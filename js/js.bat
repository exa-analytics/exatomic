REM Helper script for development on windows
call npm install
call jupyter nbextension install --py --sys-prefix exatomic
call jupyter nbextension enable --py --sys-prefix exatomic
