@echo off
chcp 65001 >nul
setlocal EnableExtensions

rem ---- 模式二：上一步把自己再拉起来，只负责"轮询健康 + 开浏览器" ----------
rem  为什么要自我复用：这样不需要多一个脚本/依赖，也不会在"服务还没起来"
rem  的时候就把浏览器打开、让用户看见一个连接失败页。
rem  这个分支必须放在 title/pushd 之前：帮手模式不碰工作目录，也不改窗口标题。
if /i "%~1"=="--wait-open" goto wait_open

title kubedocs 一键启动

rem ===========================================================================
rem  kubedocs 一键启动（Windows）
rem
rem  双击本文件即可：找到/起好服务 -> 等服务真的健康 -> 自动打开落地页。
rem  落地页按域定：clinic（默认）是问诊页 /consult，其它域是控制台 /console。
rem  起服务的那个窗口就是它的控制台，日志直接打在这里；Ctrl+C 或关窗口即停止。
rem  不写日志文件、不装任何东西、不碰 C 盘。
rem
rem  可选参数（不传就是"一键"的默认形态）：
rem    start.cmd --domain clinic      换域（默认 clinic：医疗问诊）
rem    start.cmd --domain k8s_docs    换到 K8s 文档域
rem    start.cmd --port 9000          指定端口（默认从 8000 起找第一个空闲的）
rem    start.cmd --host 0.0.0.0       换监听地址（默认 127.0.0.1，只给本机用）
rem    start.cmd --no-browser         不自动开浏览器
rem    start.cmd --online             不用 --hf-offline（默认离线，见下方说明）
rem    start.cmd --force              忽略已在运行的实例，坚持另起一个本地进程
rem
rem  为什么默认离线：模型已缓存在 HF 缓存里，联网校验反而会卡住启动。
rem  缓存不在时请改用 --online，或先跑一次 ingest 把模型拉下来。
rem ===========================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
pushd "%ROOT%"

set "DOMAIN="
set "WANTPORT="
set "WANTHOST="
set "NOBROWSER="
set "FORCE="
set "OFFLINE=--hf-offline"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--domain" goto p_domain
if /i "%~1"=="--port" goto p_port
if /i "%~1"=="--host" goto p_host
if /i "%~1"=="--no-browser" goto p_nobrowser
if /i "%~1"=="--online" goto p_online
if /i "%~1"=="--force" goto p_force
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--help" goto usage
echo [错误] 未知参数：%~1
goto usage

:p_domain
set "DOMAIN=%~2"
shift
shift
goto parse
:p_port
set "WANTPORT=%~2"
shift
shift
goto parse
:p_host
set "WANTHOST=%~2"
shift
shift
goto parse
:p_nobrowser
set "NOBROWSER=1"
shift
goto parse
:p_online
set "OFFLINE="
shift
goto parse
:p_force
set "FORCE=1"
shift
goto parse

:parsed
if "%DOMAIN%"=="" set "DOMAIN=clinic"
set "PAGE=/console"
if /i "%DOMAIN%"=="clinic" set "PAGE=/consult"
if "%WANTHOST%"=="" set "WANTHOST=127.0.0.1"

echo.
echo   kubedocs 一键启动
echo   ---------------------------------------------------------------
echo   仓库    %ROOT%
echo   域      %DOMAIN%
echo   ---------------------------------------------------------------

rem ---- 已经有服务在跑吗？----------------------------------------------------
rem  默认"附着"到已有实例，而不是再起一个。理由是本机实测到的：
rem  主机进程在提交内存紧张时会以 os error 1455（paging file too small）
rem  退化到纯全文检索，而容器里两路都正常（overlap 13/31，无降级）。
rem  再起一个本地实例只会多占内存、还拿到更差的结果。想强制另起：--force。
if not "%FORCE%"=="" goto preflight
call :probe_first
if not errorlevel 1 goto attach

rem ---- 前置检查 1：解释器 ----------------------------------------------------
:preflight
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" goto no_venv

rem ---- 前置检查 2：索引 ------------------------------------------------------
rem  索引缺失**不阻止启动**：/healthz 按设计不会因为索引空而失败，控制台页面
rem  自己会亮红条说"索引为空，先 ingest"。这里只负责把话讲清楚。
rem  下面探的是**本地回退索引**（sqlite-vec/sqlite 的位置）。默认后端 pgvector 把索引
rem  放在 Postgres 里，那时这个文件不存在是正常的——所以提示文案里说明了这一点。
if not exist "%ROOT%\data\index\%DOMAIN%\kubedocs.db" goto no_index
goto pick

:no_index
echo.
echo   [注意] 没有找到 %DOMAIN% 的本地索引：
echo          %ROOT%\data\index\%DOMAIN%\kubedocs.db
echo          用默认的 pgvector 后端时索引在 Postgres 里，这条可以忽略；
echo          否则服务仍会启动，但检索/问答会是空的。
echo          建索引（本域首次约需几分钟）：
echo            .venv\Scripts\python.exe -m kubedocs.cli --domain %DOMAIN% %OFFLINE% ingest
echo.
goto pick

rem ---- 挑端口 ----------------------------------------------------------------
rem  8000 是本机约定端口；被别的程序占着时不是"报错"而是"往后找"，直到 8019。
:pick
if not "%WANTPORT%"=="" goto have_port
set /a P=8000
:pick_loop
netstat -ano -p tcp | findstr /R /C:":%P% " | findstr /I "LISTENING" >nul 2>&1
if not errorlevel 1 goto pick_busy
set "PORT=%P%"
goto have_port
:pick_busy
echo   [信息] 端口 %P% 已被占用，试下一个。
set /a P+=1
if %P% GTR 8019 goto no_port
goto pick_loop

:have_port
rem  只有显式 --port 时才覆盖刚挑好的端口。写成无条件的 set 会在自动挑端口
rem  那条路径上把 PORT 清成空串（WANTPORT 此时是空的），服务就起不来了。
if "%WANTPORT%"=="" goto start_it
set "PORT=%WANTPORT%"
:start_it

echo   控制台  http://127.0.0.1:%PORT%/console
echo   演示页  http://127.0.0.1:%PORT%/
echo   接口文档 http://127.0.0.1:%PORT%/docs
echo   ---------------------------------------------------------------
echo   打开页面：http://127.0.0.1:%PORT%%PAGE%
echo   停止服务：在这个窗口按 Ctrl+C
echo.

rem ---- 叫醒"等待并开页面"的帮手 ---------------------------------------------
rem  用 /b：不另开窗口，和服务的日志共用这一个窗口。/b 的进程随本窗口一起结束。
rem  帮手模式自己一行都不输出，所以这里不需要重定向。
if not "%NOBROWSER%"=="1" start "" /b cmd /c ""%~f0" --wait-open %PORT% %PAGE%

rem ---- 前台跑服务 ------------------------------------------------------------
rem  前台而不是后台：Ctrl+C 和关窗口都能真的停掉它，不会留下一个孤儿进程
rem  占着端口（"窗口关了服务还在跑"这类问题在这里被结构性地排除掉）。
"%PY%" -m kubedocs.cli --domain %DOMAIN% serve --host %WANTHOST% --port %PORT% %OFFLINE%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto serve_failed
echo.
echo   服务已停止。
goto bye

:serve_failed
echo.
echo   [错误] 服务异常退出，退出码 %RC%。上面的日志是唯一线索。
pause
goto bye

rem ===========================================================================
rem  附着到已在运行的实例
rem ===========================================================================
:attach
echo.
echo   已经在 %FOUND_PORT% 上跑着一个 kubedocs 服务 —— 直接用它，不再另起一个。
echo     控制台  http://127.0.0.1:%FOUND_PORT%/console
echo     演示页  http://127.0.0.1:%FOUND_PORT%/
echo     将打开  http://127.0.0.1:%FOUND_PORT%%PAGE%
echo.
echo   想强制另起一个本地实例：start.cmd --force
echo.
if "%NOBROWSER%"=="1" goto bye_ok
start "" "http://127.0.0.1:%FOUND_PORT%%PAGE%"
:bye_ok
set "RC=0"
goto bye

rem  探 8000：上面已经有**可用**的 kubedocs 才返回 0，并把端口放进 FOUND_PORT。
rem  两个条件缺一不可：
rem    1) /healthz 的 JSON 里有 n_chunks —— 确认这是我们自己的服务，而不是别的
rem       程序碰巧占着 8000；
rem    2) n_chunks 必须**非零** —— 实测踩到过：容器在跑、但它的 pgvector 后端是
rem       空表（KUBEDOCS_DATABASE_URL 指向一个刚建好、还没灌数据的库），于是
rem       /healthz 返回 n_chunks:0。这种实例"活着但什么也搜不到"，附着上去比
rem       另起一个本地实例更糟——所以要求索引非空才算可用。
:probe_first
set "FOUND_PORT="
call :probe 8000
if errorlevel 1 exit /b 1
set "FOUND_PORT=8000"
exit /b 0

:probe
curl -s --max-time 2 "http://127.0.0.1:%~1/healthz" | findstr /R /C:"n_chunks.: *[1-9]" >nul 2>&1
exit /b %ERRORLEVEL%

rem ===========================================================================
rem  帮手模式：等服务健康后打开落地页（clinic 域是 /consult，其余是 /console）
rem ===========================================================================
:wait_open
set "WPORT=%~2"
set "WPAGE=%~3"
if "%WPAGE%"=="" set "WPAGE=/console"
set /a TRIES=0
where curl >nul 2>&1
if errorlevel 1 goto wo_sleepy

:wo_loop
set /a TRIES+=1
curl -s -o NUL --max-time 2 "http://127.0.0.1:%WPORT%/healthz" >nul 2>&1
if not errorlevel 1 goto wo_open
if %TRIES% GEQ 60 goto wo_giveup
call :sleep
goto wo_loop

rem  没有 curl 就退化成"等几秒再开"：老系统上仍然是一键，只是笨一点。
:wo_sleepy
set /a TRIES+=1
if %TRIES% GEQ 6 goto wo_open
call :sleep
goto wo_sleepy

:wo_open
start "" "http://127.0.0.1:%WPORT%%WPAGE%"
endlocal & exit /b 0

rem  一直没健康起来：安静退出。服务自己的报错在**主窗口**里，那才是该看的地方；
rem  这里再喊一声只会让用户去看一个空窗口。
:wo_giveup
endlocal & exit /b 1

rem  ping 当 sleep 用：不依赖 timeout 的交互式控制台，任何环境都能睡。
:sleep
ping -n 2 127.0.0.1 >nul
exit /b 0

rem ===========================================================================
rem  统一出口
rem ===========================================================================
rem  endlocal 和 exit 必须写在同一行：分成两行时 %RC% 会在 endlocal **之后**
rem  才展开，而那时它已随局部环境一起被丢掉——退出码永远是空的。
:bye
popd
endlocal & exit /b %RC%

:no_venv
set "RC=2"
echo.
echo   [错误] 没找到虚拟环境解释器：
echo          %PY%
echo.
echo   先建环境（只需一次）：
echo     py -3.10 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e ".[dev]"
echo.
pause
goto bye

:no_port
set "RC=3"
echo.
echo   [错误] 8000-8019 全被占用。用 --port 指定一个别的端口。
echo.
pause
goto bye

:usage
set "RC=4"
echo.
echo   用法：start.cmd [--domain 域] [--port 端口] [--host 地址]
echo                    [--no-browser] [--online] [--force]
echo.
goto bye
