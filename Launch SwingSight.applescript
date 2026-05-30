on run
    set appPath to POSIX path of (path to me)
    set projectDir to do shell script "app=" & quoted form of appPath & "; while [ \"$app\" != \"/\" ] && [ \"${app%.app}\" = \"$app\" ]; do app=$(dirname \"$app\"); done; if [ \"$app\" = \"/\" ]; then dirname " & quoted form of appPath & "; else dirname \"$app\"; fi"
    set cmd to "cd " & quoted form of projectDir & "; if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python3 app.py; exit"

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
    open location "http://127.0.0.1:8000"
end run
