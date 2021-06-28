# 멀티에이전트 기반 AI worldcup 축구 에이전트 설계
---
## Random walk
> 로봇의 양측 바퀴 속도를 random으로 결정하는 알고리즘으로, 간단히 말하면 ‘무작위 움직임’ agent  

#### 1. [`random-walk`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Random%20Walk/random-walk)
- 로봇의 좌우측 바퀴 속도를 random으로 결정하여 무작위로 움직이는 Strategy

---
## Dynamic planning
> Rule-Based 알고리즘을 기반으로 하며, 확고한 규칙을 가진 전략을 세우고 그에 따라 행동하는 agent

#### 1. [`Standard role-based players`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Standard_role-based_players)
- 공의 위치와 로봇의 위치에 따른 현재 상태를 기반으로 알고리즘을 설정하여 미리 설정된 규칙대로 움직이는 Strategy


#### 2. [`Gegenangriff`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Gegenangriff) - becoming extremely offensive players
- GK를 제외한 수비수 2명, 공격수 2명이 모두 Field에서만 공을 몰고가서 골을 넣는 Strategy
- **Issue**
  - 공이 경기장 구석으로 가면 공을 감싸서 정체되는 시간이 발생
  - 네 명이 전부 달려들다 보니 같은 팀 인데도 서로의 진로를 방해하는 경우가 발생
  - 공이 우리 팀 진영으로 가고있을 때는 방향을 꺾어서 상대팀 진영으로 공을 끌고가야 되는데, 네 명 모두 방향을 바꾸려다 보니 우리 팀 진영으로 몰고가는 경우가 발생


#### 3. [`Gegenpressing`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Gegenpressing) – becoming extremely defensive players
- 우리 팀 수비수 2명이 상대팀 공격수 방어, 우리 공격수 1명이 상대팀 수비수 방어, 우리 팀 나머지 공격수가 상대팀 골대에 골을 넣는 Strategy
- **Issue**
  - 경로가 겹치는 경우가 존재
  - 상대 수비수도 공격하는 경우가 존재
  - 우리 팀 수비수가 없어서 공격을 막을 로봇이 없음
  - 상대팀 공격수를 쫓아다니다가 자책골을 자주 만듦
  - F2가 공을 따라가다가 다른 로봇에 걸린 경우, 공을 따라가지 못함	


#### 4. [`Becoming sprinters`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming_sprinters)
- 좀 더 빠른 속력을 갖는 Strategy
- **Issue**
  - 우리 팀이 상대 팀을 빠릿빠릿하게 따라가지 못함
  - 우리 팀 D2가 상대팀 D2를 따라가는 도중 상대팀 F2와 충돌하게 됨
  - 상대팀 F2를 따라가던 우리 팀 F2까지 세 개의 로봇이 몸싸움을 하다가 우리 팀 D2와 F2가 서로의 갈 길을 막아 상대팀을 빠르게 따라가지 못하고 뒤늦게 쫓아가는 상황 발생
  - 공의 위치에 따라 상대팀 F1의 위치도 달라지기 때문에 위의 두 상황에서 F1의 행동이 똑같다고는 말할 수 없지만,여러 번 시뮬레이션을 돌려본 결과 로봇들이 뭉쳐서 몸싸움을 진행해도 비교적 빠르게 그 상황을 벗어남


#### 5. [`Becoming defenders in the penalty area`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming%20defenders%20in%20the%20penalty%20area)
- 우리 팀 골대를 Penalty Area 바로 앞 바깥 영역에서 방어하는 Strategy
- **Issue**
  - 골을 막다가 특수한 상황(PENALTYKICK, KICKOFF, CORNERKICK 등)이 발생하기도 해서 수비에 방해
  - 수비수가 수비를 하다 Penalty area에 들어가는 경우 Penalty Kick이 발생
  - 공이 수비 영역 내 있지 않으면 Delay가 발생
  - 해당 영역이 아닌 위치에서 수비를 진행하게 되어 수비 범위 내 이동이 자유롭지 못함


#### 6. [`Becoming man-to-man defenders`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming%20man-to-man%20defenders)
- 상대 팀을 따라다니는 Strategy
- **Issue**
  - 골키퍼가 Penalty area를 3초 이상 벗어나면 default position으로 돌아간다는 규칙때문에 상대팀(Red) 골키퍼에게 가지 못하고 계속 되돌아옴 `-> 원래 위치에서 기존 역할을 수행하게 함`

#### 7. [`Becoming ball-hunters`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming_ball-hunters)
- 무조건 공만 따라다니는 Strategy
- **Issue**
  - 골키퍼가 Penalty area 을 3초 이상 벗어나면 제자리로 돌아옴
  - 골키퍼 가운데 가만히 있는 경우 계속 튕겨져 나옴
  - 골키퍼를 Penalty area 끝으로 이동시킴
  - 공을 따라다니기만 하면, 구석에서 Delay 발생
  - 공 따라다니면서 점유 시 킥을 하는 경우, 맨 처음 reload하면 무조건 1골 먹힘


#### 8. [`All Becoming goalies #1`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/All%20Becoming%20goalies_1)
- 우리 팀 5명이 상대 팀 골대에 위치하여 상대방 골대를 막고, 우리 골대는 비워 두는 Strategy
- **Issue**
  - 우리 팀 5명 모두 상대팀 골대 앞에 위치하지 않음
  - 상대팀 골대로 가는 도중에 골이 먹히는 경우도 발생
  - 우리 팀 Agent들이 상대 Penalty area로 들어가면 다시 원래 위치로 돌아 감


#### 9. [`All Becoming goalies #2`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/All%20Becoming%20goalies_2)  
- 우리 팀 골대 앞에 일렬로 서서 수비하는 Strategy
- **Issue**
  - 예상 결과인 ‘상대팀이 유효 슛 시도를 해도 골이 먹히지 않음’과는 다르게, 실제 결과는 ‘계속 Penalty Kick만 반복되고, 골이 먹힘’과 같이 나옴


---
## Multi-agent Reinforence Learning
> IQL과 QMIX 두 가지 알고리즘을 사용하며, random walk와 여러 규칙 기반 agent들을 상대로 학습시켜 모델을 얻는 agent
