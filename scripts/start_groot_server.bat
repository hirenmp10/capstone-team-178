@echo off
rem Start the GR00T policy server with its environment scoped to the server
rem process only. All arguments are passed to start_groot_server.ps1, e.g.
rem     scripts\start_groot_server.bat -DryRun
rem     scripts\start_groot_server.bat -Port 5555 -Device cuda:0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_groot_server.ps1" %*
exit /b %ERRORLEVEL%
