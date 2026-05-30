on run
    set appPath to POSIX path of (path to me)
    set projectDir to do shell script "dirname " & quoted form of appPath
    set cmd to "cd " & quoted form of projectDir & "; if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python app.py; exit"

    tell application "Terminal"
        activate
        do script cmd
    end tell

    delay 1.5
    open location "http://127.0.0.1:8000"
end run
