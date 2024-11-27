# string_id_1=70010
# string_id_2=70020
# string_id_3=70030 
string_id_4=71040

# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --task_string $string_id_1 > $string_id_1'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_1.pid
# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --use_deep_learner_outcome --task_string $string_id_1 > $string_id_1'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_1.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --task_string $string_id_2 > $string_id_2'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_2.pid
# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --use_deep_learner_outcome --task_string $string_id_2 > $string_id_2'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_2.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --task_string $string_id_3 > $string_id_3'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_3.pid
# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --use_deep_learner_outcome --task_string $string_id_3 > $string_id_3'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_3.pid  

# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --task_string $string_id_4 > $string_id_4'_'training_outcomeLR.log 2>&1 & echo $! > command'_'$string_id_4.pid
nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --task_string $string_id_4 > $string_id_4'_'training_outcomeLR_0.log 2>&1 & echo $! > command'_'$string_id_4'_0'.pid

# nohup python DEV_dl_generalized_quarantine_time_series_dl_copy.py --use_deep_learner_outcome --task_string $string_id_4 > $string_id_4'_'training_outcomeDL.log 2>&1 & echo $! > command_dl'_'$string_id_4.pid  