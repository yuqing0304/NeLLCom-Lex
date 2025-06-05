#### Overall experiments structure
##### 3 training pipelines* 
###### (a) SL w/ context   + RL w/ context (zoo/color_game, turn on --if_context flag)
###### (b) SL w/o context + RL w/ context (zoo/color_gamezero, which means speakers' parameters are zeroed out during SL)
###### (c) SL w/o context + RL w/o context (zoo/color_game)
###### * SL on human data; RL on generated data w/ human-like proportion

#### Using pipeline (a) train different RL data distributions in terms of close/far: 
###### 100/0, 50/50, 0/100
###### Agents are evaluated on generated data w/ 50/50 close/far proportion

