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

#### 3. [`Gegenpressing`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Gegenpressing) – becoming extremely defensive players
- 우리 팀 수비수 2명이 상대팀 공격수 방어, 우리 공격수 1명이 상대팀 수비수 방어, 우리 팀 나머지 공격수가 상대팀 골대에 골을 넣는 Strategy

#### 4. [`Becoming sprinters`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming_sprinters)
- 좀 더 빠른 속력을 갖는 Strategy

#### 5. [`Becoming defenders in the penalty area`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming%20defenders%20in%20the%20penalty%20area)
- 우리 팀 골대를 Penalty Area 바로 앞 바깥 영역에서 방어하는 Strategy

#### 6. [`Becoming man-to-man defenders`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming%20man-to-man%20defenders)
- 상대 팀을 따라다니는 Strategy

#### 7. [`Becoming ball-hunters`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/Becoming_ball-hunters)
- 무조건 공만 따라다니는 Strategy

#### 8. [`All Becoming goalies #1`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/All%20Becoming%20goalies_1)
- 우리 팀 5명이 상대 팀 골대에 위치하여 상대방 골대를 막고, 우리 골대는 비워 두는 Strategy

#### 9. [`All Becoming goalies #2`](https://github.com/I-hate-Soccer/AI_Soccer/tree/main/Dynamic%20Planning/All%20Becoming%20goalies_2)  
- 우리 팀 골대 앞에 일렬로 서서 수비하는 Strategy

---
## Multi-agent Reinforence Learning
> IQL과 QMIX 두 가지 알고리즘을 사용하며, random walk와 여러 규칙 기반 agent들을 상대로 학습시켜 모델을 얻는 agent
