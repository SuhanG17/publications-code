string_id_1=60010
string_id_2=60020
string_id_3=60030
string_id_4=60040

# nohup python DEV_dl_generalized_quarantine_time_series_dl.py --task_string $string_id_1 > $string_id_1'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_1.pid
nohup python DEV_dl_generalized_quarantine_time_series_dl.py --use_deep_learner_outcome --task_string $string_id_1 > $string_id_1'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_1.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl.py --task_string $string_id_2 > $string_id_2'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_2.pid
nohup python DEV_dl_generalized_quarantine_time_series_dl.py --use_deep_learner_outcome --task_string $string_id_2 > $string_id_2'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_2.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl.py --task_string $string_id_3 > $string_id_3'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_3.pid
nohup python DEV_dl_generalized_quarantine_time_series_dl.py --use_deep_learner_outcome --task_string $string_id_3 > $string_id_3'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_3.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl.py --task_string $string_id_4 > $string_id_4'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_4.pid
nohup python DEV_dl_generalized_quarantine_time_series_dl.py --use_deep_learner_outcome --task_string $string_id_4 > $string_id_4'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_4.pid  