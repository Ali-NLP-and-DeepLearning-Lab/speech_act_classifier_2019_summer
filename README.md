# Speech act recognition

Set up any virtual environment with python 3.6.1 and at '{your_dir}/VRM_recognition/' type following install command.

    pip install -r requirements.txt 

If you don't have 'log', 'checkpoint', 'data' folder in '{your_dir}/VRM_recognition/', please make them.

    mkdir log
    mkdir checkpoint
    mkdir data

You can run training by below commands. each for SwDA and VRM respectfully.

    sh run_train_swda.sh
    sh run_train_vrm.sh

Training log will be saved in 'log/' you can check it by tensorboard. 

    ## at server side
    cd log
    tensorboard --logdir ./
    
    ## at host side below is example for using kixlab server
    ssh -L 16006:127.0.0.1:6006 kixlab2@internal.kixlab.org -p 22   
    go https0.0.0.0:16006 address in you browser
    
You can run validation by below commands. 
    
    sh run_val_swda.sh
    sh run_val_vrm.sh

It will print precision for speech tag in swda and for vrm, it will print precision for each axis. 
and also for swda, validation will save result of validation, csv files with 'conversation_id', 'utterance_id', 'gt', 'pred'.
the default saving path is 'data/BiLSTM-RAM_SWDA_comm_output.csv' then by running following script, it will compute precision for each tag and save csv file with utterance and ground truth tag and predicted tag. only swda supported.

    sh run_analysis.sh
    
Please check PATH configuration in scripts file carefully. for validation and analysis, exact file path for model checkpoint or output file should be typed in script file. 

what you need in '{your_dir}/data/' is following files.

    'swda_train_data.csv'
    'swda_val_data.csv'
    'vrm_train_data.csv'
    'vrm_val_data.csv'
    'word2vec_from_glove.bin'
    'GoogleNews-vectors-negative300.bin'  -> optional. not used. using glove one instead

For data processing, for SwDA I referred 'https://github.com/SeanCherngTW/SwDA-Preprocessing' and for VRM I left 'src/utils/vrm_preprocess.py'. please check. (not for running in this folders.)

You can check model architecture in 'src/model/bilstm_ram.py'. and configuration information is in files in 'src/config/'.

Data pipeline is implemented in 'src/dataset/'. the code is quite complicated and it is tightly correlated with format of data, so please if you want to modify it, check train_data.csv file carefully. or also there is a simple version of data preprocessing in 'sar_endpoint/speech_act/views.py', so please check.

# Speech-act recognition endpoint

demo endpoint is implemented by django. 

    cd sar_endpoint
    python manage.py migrate
    python manage.py runserver
    
    # and you can check demo by following code in other cmd
    python sar_endpoint/test.py

this is tested in server itself. maybe you should change ip setting in 'sar_endpoint/sar_endpoint/settings.py' ALLOWED_HOSTS or in test.py. 

there is runnable end point in 'check_vrm/' folder in 'kixlab2' server. please activate 'vrm-torch' virtual environment and follow above instructions.
