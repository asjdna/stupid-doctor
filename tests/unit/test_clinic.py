"""clinic 域 P1 测试：急诊红线安全门控 + 号源并发不超卖。

两组都是安全相关的，断言写得比一般单测更死：红线误报会把家属、既往或否认的症状
当成患者本人的急诊，号源超卖会让两个人拿到同一个号。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from kubedocs.domains.clinic.booking import BookingService, describe_ticket
from kubedocs.domains.clinic.mock import MockAppointmentProvider
from kubedocs.domains.clinic.polarity import (
    assertion,
    is_negated,
    is_reportable,
    polarity,
    status_of,
)
from kubedocs.domains.clinic.provider import BookingError
from kubedocs.domains.clinic.redline import (
    EMERGENCY_RED_LINE,
    RiskGuard,
    emergency_advice,
    emergency_menu,
    wants_emergency_care,
)
from kubedocs.domains.clinic.sqlite_provider import SqliteAppointmentProvider


# 一、ConText 三维：急诊红线安全性的根基
class TestPolarity:
    def test_plain_negation(self):
        assert is_negated("我没有胸痛", "胸痛") is True
        assert polarity("我没有胸痛", "胸痛") == "absent"

    def test_present_is_not_negated(self):
        assert is_negated("我现在胸痛", "胸痛") is False

    def test_scope_stops_at_reversal(self):
        """否定作用域到转折词为止：「无发热，但是咳嗽」里咳嗽是现症。"""
        assert is_negated("无发热，但是咳嗽", "发热") is True
        assert is_negated("无发热，但是咳嗽", "咳嗽") is False

    def test_fake_negation_compound(self):
        """「无痛性血尿」里「无痛」是修饰，血尿仍在——不能判成否认。"""
        assert is_negated("无痛性血尿", "血尿") is False

    def test_non_negation_compound(self):
        """「没劲」本身是症状（乏力），其中的「没」不该吞掉后面的症状。"""
        assert is_negated("没劲犯困", "犯困") is False

    def test_uncertain_is_possible_not_absent(self):
        """「不能排除胸痛」是"要警惕"，不是"没有胸痛"。"""
        assert status_of("不能排除胸痛", "胸痛") == "possible"
        assert is_negated("不能排除胸痛", "胸痛") is False

    def test_hypothetical(self):
        assert status_of("如果胸痛的话怎么办", "胸痛") == "hypothetical"

    def test_all_occurrences_rule(self):
        """只要有一处不是否认，就不能判否认（医学安全偏向"宁可留着"）。"""
        # 分号是硬边界，否定作用域不跨越它
        assert is_negated("没有胸痛；胸痛", "胸痛") is False

    def test_comma_does_not_end_negation_scope(self):
        """已知局限：逗号不截断否定作用域。

        逗号不截断是有意的——「无发热、咳嗽」这类**列举**里截断会把咳嗽误判成现症；
        代价是「我以前没有胸痛，现在胸痛」也被判成否认，因为 `现在` 这类时间重启词
        不在 `REVERSAL_WORDS` 里。本测试固定这个行为，避免它被无声改掉。
        """
        assert is_negated("无发热、咳嗽", "咳嗽") is True  # 列举：不该截断
        assert is_negated("我以前没有胸痛，现在胸痛", "胸痛") is True  # 已知局限
        assert polarity("我以前没有胸痛，现在胸痛", "胸痛") == "absent"

    @pytest.mark.parametrize(
        ("text", "phrase"),
        [
            ("无腹泻，其实是肚子疼", "肚子疼"),
            ("无胸痛、无呼吸困难，仅有轻微头晕", "头晕"),
        ],
    )
    def test_reversal_words_cut_the_scope(self, text, phrase):
        """转折词 `其实` / `仅` 必须截断否定作用域。

        缺这两个词时作用域会越过逗号盖住后半句，把"肚子疼""头晕"判成**否认**；
        注意 `现在` 这类时间重启词不在此列，上面那条已知局限依然存在。
        """
        assert polarity(text, phrase) == "present"
        assert is_reportable(text, phrase) is True


class TestPolaritySafetyGate:
    """`is_reportable` 是急诊红线唯一的安全阀，四类必须挡住。"""

    def test_self_present_is_reportable(self):
        assert is_reportable("我现在胸痛", "胸痛") is True
        assert is_reportable("胸痛，喘不上气", "胸痛") is True
        assert is_reportable("胸痛", "胸痛") is True  # 缺省即本人/现在

    def test_other_person_is_not_reportable(self):
        """核心用例：家属的症状不能算患者本人的主诉。"""
        text = "我老公胸痛，我没事"
        assert is_reportable(text, "胸痛") is False
        assert assertion(text, "胸痛")["experiencer"] == "other"

    def test_past_is_not_reportable(self):
        text = "我去年胸痛过一次，现在好了"
        assert is_reportable(text, "胸痛") is False
        assert assertion(text, "胸痛")["temporality"] == "past"

    def test_negated_is_not_reportable(self):
        assert is_reportable("我没有胸痛", "胸痛") is False

    def test_hypothetical_is_not_reportable(self):
        assert is_reportable("如果胸痛怎么办", "胸痛") is False

    def test_own_symptom_after_relative_is_kept(self):
        """「我父亲得过心梗，我胸闷」：患者自己的胸闷不能被前文带成家属的或既往的。

        `is_reportable` 要求短语字面出现在文本里，所以这里不能改述成"胸口有点闷"。
        """
        text = "我父亲得过心梗，我胸闷"
        assert assertion(text, "胸闷")["experiencer"] == "self"
        assert assertion(text, "胸闷")["temporality"] == "present"
        assert assertion(text, "胸闷")["status"] == "present"
        assert is_reportable(text, "胸闷") is True

    def test_phrase_absent_from_text_is_not_reportable(self):
        """短语不在文本里时一律"不可上报"——这正是 agent 侧需要症状别名表的原因。"""
        assert assertion("我胸口有点闷", "胸闷")["status"] == "absent"
        assert is_reportable("我胸口有点闷", "胸闷") is False


# 二、急诊红线
class TestRiskGuard:
    def setup_method(self):
        self.guard = RiskGuard()

    @pytest.mark.parametrize(
        "text",
        [
            "我现在胸痛",
            "胸痛，喘不上气",
            "突然就晕倒了",
            "半边身子不能动",
        ],
    )
    def test_red_lines_trigger(self, text):
        hit = self.guard.check(text)
        assert hit is not None, f"漏掉急诊：{text}"

    @pytest.mark.parametrize(
        "text,why",
        [
            ("我老公胸痛，我没事", "他人家属"),
            ("我去年胸痛过一次，现在好了", "既往"),
            ("我没有胸痛，就是有点闷", "否认"),
            ("如果胸痛的话要不要去医院", "假设"),
        ],
    )
    def test_must_not_trigger(self, text, why):
        hit = self.guard.check(text)
        assert hit is None, f"误报急诊（{why}）：{text} -> {hit.describe() if hit else ''}"

    def test_hit_carries_the_three_dimensions(self):
        hit = self.guard.check("我现在胸痛")
        assert hit is not None
        assert (hit.status, hit.experiencer, hit.temporality) == ("present", "self", "present")

    def test_explain_lists_all_red_line_words_with_verdicts(self):
        """诊断能力是必需的：只看布尔结果无法回答"为什么这条没触发"。"""
        lines = self.guard.explain("我老公胸痛，我没事")
        assert any(line.startswith("不触发") and "胸痛" in line for line in lines)

    def test_red_line_word_list_is_not_empty_and_has_no_duplicates(self):
        assert len(EMERGENCY_RED_LINE) >= 10
        assert len(set(EMERGENCY_RED_LINE)) == len(EMERGENCY_RED_LINE)


class TestEmergencyCopy:
    def test_menu_offers_two_choices_and_does_not_force(self):
        text = emergency_menu("胸痛")
        assert "①" in text and "②" in text
        assert "120" in text

    def test_advice_mentions_120(self):
        assert "120" in emergency_advice("胸痛")

    @pytest.mark.parametrize("text", ["打120", "叫救护车", "我去急诊", "马上到医院"])
    def test_choosing_care_is_detected(self, text):
        assert wants_emergency_care(text) is True

    @pytest.mark.parametrize("text", ["先不去了", "继续问问看"])
    def test_choosing_to_continue(self, text):
        assert wants_emergency_care(text) is False


# 三、挂号：契约 + Mock 替身
class TestMockProviderContract:
    def test_departments_have_code_name_fee(self):
        deps = MockAppointmentProvider().list_departments()
        assert deps and {"code", "name", "fee"} <= set(deps[0])

    def test_availability_then_book_decrements(self):
        p = MockAppointmentProvider()
        before = {s["time"]: s["remaining"] for s in p.get_availability("心内科")}
        ticket = p.book("心内科", "u1", time_slot="上午")
        after = {s["time"]: s["remaining"] for s in p.get_availability("心内科")}
        assert after["上午"] == before["上午"] - 1
        assert ticket["queue_no"] == 1

    def test_sold_out_raises_business_error(self):
        p = MockAppointmentProvider()
        p.set_remaining("心内科", "上午", 1)
        p.book("心内科", "u1", time_slot="上午")
        with pytest.raises(BookingError, match="挂完"):
            p.book("心内科", "u2", time_slot="上午")

    def test_bad_slot_is_refused(self):
        with pytest.raises(BookingError, match="时段"):
            MockAppointmentProvider().book("心内科", "u1", time_slot="半夜")

    def test_fault_injection(self):
        p = MockAppointmentProvider(failing_departments=("心内科",))
        with pytest.raises(BookingError, match="暂时不可用"):
            p.get_availability("心内科")

    def test_list_appointments_is_per_patient_and_newest_first(self):
        p = MockAppointmentProvider()
        p.book("心内科", "u1", time_slot="上午")
        p.book("呼吸内科", "u1", time_slot="下午")
        p.book("皮肤科", "u2", time_slot="上午")
        mine = p.list_appointments("u1")
        assert len(mine) == 2
        assert all(a["patient_id"] == "u1" for a in mine)


class TestBookingService:
    def test_resolve_department_or_refuse_to_guess(self):
        svc = BookingService(MockAppointmentProvider())
        assert svc.resolve_department("心内科")
        with pytest.raises(BookingError, match="没能识别"):
            svc.resolve_department("")

    def test_unknown_department_lists_alternatives(self):
        svc = BookingService(MockAppointmentProvider())
        with pytest.raises(BookingError, match="不在可挂号科室"):
            svc.resolve_department("火星科")

    def test_book_picks_first_available_slot(self):
        svc = BookingService(MockAppointmentProvider())
        ticket = svc.book("心内科", "u1")
        assert ticket["time"] in ("上午", "下午")
        assert ticket["ticket_no"]

    def test_book_when_all_slots_exhausted(self):
        p = MockAppointmentProvider()
        p.set_remaining("心内科", "上午", 0)
        p.set_remaining("心内科", "下午", 0)
        svc = BookingService(p)
        with pytest.raises(BookingError, match="没有余号"):
            svc.book("心内科", "u1")

    def test_describe_ticket_is_human_readable(self):
        svc = BookingService(MockAppointmentProvider())
        text = describe_ticket(svc.book("心内科", "u1"))
        assert "挂号成功" in text and "排队号" in text


# 四、SQLite 号源库：真实持久化 + 并发安全
@pytest.fixture
def sqlite_provider(tmp_path: Path):
    p = SqliteAppointmentProvider(tmp_path / "clinic.db")
    yield p
    p.close()


class TestSqliteProvider:
    def test_schema_and_constraints_exist(self, sqlite_provider: SqliteAppointmentProvider):
        con = sqlite_provider._con
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"patients", "schedules", "appointments"} <= tables
        triggers = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert any("fk" in t for t in triggers)

    def test_booking_persists_across_reopen(self, tmp_path: Path):
        db = tmp_path / "p.db"
        with SqliteAppointmentProvider(db) as p:
            ticket = p.book("心内科", "u1", patient_name="张三", time_slot="上午")
        with SqliteAppointmentProvider(db) as p2:
            mine = p2.list_appointments("u1")
        assert [m["ticket_no"] for m in mine] == [ticket["ticket_no"]]

    def test_remaining_decreases_and_is_visible(self, sqlite_provider):
        before = {s["time"]: s["remaining"] for s in sqlite_provider.get_availability("心内科")}
        sqlite_provider.book("心内科", "u1", time_slot="上午")
        after = {s["time"]: s["remaining"] for s in sqlite_provider.get_availability("心内科")}
        assert after["上午"] == before["上午"] - 1

    def test_booking_unknown_patient_is_refused_by_trigger(self, tmp_path: Path):
        """绕过 ensure_patient 直接插号：触发器必须挡住（外键 PRAGMA 是连接级的，不可靠）。"""
        p = SqliteAppointmentProvider(tmp_path / "t.db")
        try:
            p.get_availability("心内科")  # 先建号源
            with pytest.raises(sqlite3.IntegrityError, match="不存在的患者"):
                p._con.execute(
                    "INSERT INTO appointments(patient_id, schedule_id, department, date,"
                    " time_slot, ticket_no, queue_no) VALUES('ghost',1,'心内科',"
                    "'2026-01-01','上午','X1',1)"
                )
        finally:
            p.close()

    def test_remaining_cannot_go_negative(self, sqlite_provider):
        """数据库级兜底：任何写入路径都不能把余号改成负数。"""
        sqlite_provider.get_availability("心内科")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            sqlite_provider._con.execute("UPDATE schedules SET remaining = -1 WHERE department='心内科'")

    def test_duplicate_schedule_row_is_refused(self, sqlite_provider):
        sqlite_provider.get_availability("心内科")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            sqlite_provider._con.execute(
                "INSERT INTO schedules(department,date,time_slot,total,remaining)"
                " SELECT department,date,time_slot,total,remaining FROM schedules LIMIT 1"
            )

    def test_cancel_returns_the_slot(self, sqlite_provider):
        ticket = sqlite_provider.book("心内科", "u1", time_slot="上午")
        mid = {s["time"]: s["remaining"] for s in sqlite_provider.get_availability("心内科")}
        sqlite_provider.cancel(ticket["ticket_no"], "u1")
        after = {s["time"]: s["remaining"] for s in sqlite_provider.get_availability("心内科")}
        assert after["上午"] == mid["上午"] + 1

    def test_cancel_twice_is_refused(self, sqlite_provider):
        ticket = sqlite_provider.book("心内科", "u1", time_slot="上午")
        sqlite_provider.cancel(ticket["ticket_no"], "u1")
        with pytest.raises(BookingError, match="已经取消"):
            sqlite_provider.cancel(ticket["ticket_no"], "u1")

    def test_cannot_cancel_someone_elses_ticket(self, sqlite_provider):
        ticket = sqlite_provider.book("心内科", "u1", time_slot="上午")
        with pytest.raises(BookingError, match="没找到"):
            sqlite_provider.cancel(ticket["ticket_no"], "u2")

    def test_bad_date_is_refused(self, sqlite_provider):
        with pytest.raises(BookingError, match="日期格式"):
            sqlite_provider.book("心内科", "u1", date="2026/1/1", time_slot="上午")

    def test_ticket_numbers_are_unique(self, sqlite_provider):
        numbers = {sqlite_provider.book("心内科", "u1", time_slot="上午")["ticket_no"] for _ in range(5)}
        assert len(numbers) == 5

    def test_queue_numbers_are_sequential_per_department_day(self, sqlite_provider):
        a = sqlite_provider.book("心内科", "u1", time_slot="上午")
        b = sqlite_provider.book("心内科", "u2", time_slot="上午")
        c = sqlite_provider.book("心内科", "u3", time_slot="下午")
        assert [a["queue_no"], b["queue_no"], c["queue_no"]] == [1, 2, 3]


class TestNoOversellUnderConcurrency:
    """号源系统唯一不能妥协的性质：并发下不超卖。

    余号恰好 1 时 8 个线程同抢：恰好一个成功，余号归零且不会变负。
    """

    def test_exactly_one_winner_with_one_slot_left(self, tmp_path: Path):
        p = SqliteAppointmentProvider(tmp_path / "race.db")
        try:
            p.get_availability("心内科")
            p._con.execute(
                "UPDATE schedules SET remaining=1, total=1 WHERE department='心内科' AND time_slot='上午'"
            )
            barrier = threading.Barrier(8)
            results: list[str] = []
            lock = threading.Lock()

            def worker(i: int) -> None:
                barrier.wait()  # 尽量让 8 个线程同时开抢
                try:
                    p.book("心内科", f"u{i}", time_slot="上午")
                    with lock:
                        results.append("ok")
                except BookingError:
                    with lock:
                        results.append("sold_out")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert results.count("ok") == 1, f"超卖了：{results}"
            assert results.count("sold_out") == 7
            row = p._con.execute(
                "SELECT remaining FROM schedules WHERE department='心内科' AND time_slot='上午'"
            ).fetchone()
            assert row[0] == 0
            assert (
                p._con.execute("SELECT COUNT(*) FROM appointments WHERE department='心内科'").fetchone()[0]
                == 1
            )
        finally:
            p.close()

    def test_queue_numbers_unique_under_concurrency(self, tmp_path: Path):
        p = SqliteAppointmentProvider(tmp_path / "race2.db")
        try:
            p.get_availability("心内科")
            barrier = threading.Barrier(6)
            errors: list[str] = []

            def worker(i: int) -> None:
                barrier.wait()
                try:
                    p.book("心内科", f"u{i}", time_slot="上午")
                except BookingError:
                    pass
                except Exception as e:  # noqa: BLE001 - 记录任何意外异常
                    errors.append(f"{type(e).__name__}: {e}")

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"并发下出现意外异常：{errors}"
            queues = [
                r[0] for r in p._con.execute("SELECT queue_no FROM appointments WHERE department='心内科'")
            ]
            assert len(queues) == len(set(queues)), f"排队号重复：{sorted(queues)}"
        finally:
            p.close()
