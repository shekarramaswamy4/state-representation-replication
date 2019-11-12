ram_state_dict = {
    "breakout": dict(ball_x=99,
                     ball_y=101,
                     player_x=72,
                     blocks_hit_count=77,
                     block_bit_map=range(30),  # see breakout bitmaps tab
                     score=84),  # 5 for each hit
    "pong": dict(player_y=51,
                 enemy_y=50,
                 ball_x=49,
                 ball_y=54,
                 enemy_score=13,
                 player_score=14),
    "seaquest": dict(enemy_obstacle_x=range(30, 34),
                     player_x=70,
                     player_y=97,
                     diver_x=range(71, 75),
                     player_direction=86,
                     missile_direction=87,
                     oxygen_meter_value=102,
                     missile_x=103,
                     score=[57, 58],
                     num_lives=59,
                     divers_collected_count=62),
    "spaceinvaders": dict(invaders_left_count=17,
                          player_score=104,
                          num_lives=125,
                          player_x=28,
                          enemies_x=26,
                          missiles_y=9,
                          enemies_y=24)
}

update_dict = {k: {} for k in atari_dict.keys()}

remove_dict = {k: [] for k in atari_dict.keys()}

for game, d in atari_dict.items():
    for k, v in d.items():
        if isinstance(v, range) or isinstance(v, list):
            for i, vi in enumerate(v):
                update_dict[game]["%s_%i" % (k, i)] = vi
            remove_dict[game].append(k)

for k in atari_dict.keys():
    atari_dict[k].update(update_dict[k])
    for rk in remove_dict[k]:
        atari_dict[k].pop(rk)
