on run
    set appPath to POSIX path of (path to me)
    set projectDir to do shell script "app=" & quoted form of appPath & "; while [ \"$app\" != \"/\" ] && [ \"${app%.app}\" = \"$app\" ]; do app=$(dirname \"$app\"); done; if [ \"$app\" = \"/\" ]; then dirname " & quoted form of appPath & "; else dirname \"$app\"; fi"
    set pidFile to projectDir & "/.swingsight.pid"
    set cmd to "cd " & quoted form of projectDir & "; if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python3 app.py > swingsight.log 2>&1 & echo $! > " & quoted form of pidFile & "; wait; exit"

    tell application "Terminal"
        activate
        do script cmd
    end tell

    delay 1.0
    repeat 12 times
        try
            do shell script "curl -s http://127.0.0.1:8000/ >/dev/null"
            exit repeat
        on error
            delay 0.5
        end try
    end repeat
    tell application "Safari"
        activate
        open location "http://127.0.0.1:8000"
    end tell

    repeat
        delay 1
        set hasPage to false
        if application "Safari" is running then
            tell application "Safari"
                repeat with w in windows
                    repeat with t in tabs of w
                        if (URL of t) begins with "http://127.0.0.1:8000" then
                            set hasPage to true
                        end if
                    end repeat
                end repeat
            end tell
        end if
        if hasPage is false then exit repeat
    end repeat

    do shell script "if [ -f " & quoted form of pidFile & " ]; then kill $(cat " & quoted form of pidFile & "); rm -f " & quoted form of pidFile & "; fi"
end run
