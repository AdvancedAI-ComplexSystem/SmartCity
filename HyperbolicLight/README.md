Anonymous GitHub for HyperbolicLight

data:
  Hangzhou:
      anon_4_4_hangzhou_real == hangzhou1
      anon_4_4_hangzhou_real_5816 == hangzhou2
  Jinan:
      anon_3_4_jinan_real == jinan1
      anon_3_4_jinan_real_2000 == jinan2
      anon_3_4_jinan_real_2500 == jinan3
  City256:
      anon_16_16_custom_sumo_2 == city256
  NewYork:
      anon_28_7_newyork_real_double == newyork1
      anon_28_7_newyork_real_triple == newyork2

results:
  all_results for paper results

run hyperbolic DQN for TSC code:
  python run_hyperboliclight.py

run summary ATT:
  python summary.py
