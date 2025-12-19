# Trade-Builder/RL-models

- 파이썬 버전: 3.11 이상

- 사용방법:    
requirements.txt를 이용해 가상환경을 세팅하고, socket_server.py를 실행하면 됩니다.

- Trade-Builder-Client의 프로세스로 넣을 경우:   
 Trade-Builder-Client 내부의 코드를 수정해야 합니다.
 "Trade-Builder-Client/electron/rl_launcher.js" 파일을 찾아, RLCmd 변수에는 가상환경 파이썬 경로를, scriptPath 변수에는 socket_server.py 파일의 경로를 넣으면 됩니다.