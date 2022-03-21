import csv 
import pandas as pd
import numpy as np 
import os 
import shutil
import subprocess
import argparse 
import apiaudio
import os
import requests
import shutil
from urllib.request import urlretrieve

def generate_filepath(speaker, n):
    outpath = os.path.join(f"{os.getcwd()}", speaker, f"{n}.wav")
    return(outpath)

def generate_sample(script, speaker, audience):
    # utt_txt = audience
    r = apiaudio.Speech().create(
        scriptId=script.get("scriptId"),
        voice=speaker,
        silence_padding=0,
        audience=audience,
    )
    try:    
        url = r['main']['url']
        return url
    except:
        print(f"Error with the following utterance: {utt}")

    

def download_audio_file(url, filename):
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

def return_targets(filename, column_index, range_min, range_max):
    df = pd.read_csv(filename, header='infer')
    return df.loc[(df.iloc[:, column_index] >= range_min) & (df.iloc[:, column_index] <= range_max)]

def copy_files(data, file_dir, out_dir, speaker, n_files):
    for filename in data.iloc[0:n_files, 0]:
        file_loc =os.path.join(file_dir, speaker, filename)
        out_loc =os.path.join(out_dir, speaker, filename)
        shutil.copyfile(file_loc, out_loc)

def inrange_write_to_csv(filename, data, speaker, n_rows):
    # which_mode = 'a' if os.path.exists(filename) else 'w+'
    data['Voice'] = [speaker] * len(data.iloc[:,1])
    data = data.iloc[0:n_rows, [-1]+list(range(len(data.iloc[1,:-1])))] # reorders columns so name is first
    write_to_csv(filename, data)
    # return len(data.iloc[:,1])

def calculate_stats(filename, column_index):
    df = pd.read_csv(filename, header='infer')
    mean_score = np.mean(df.iloc[:, 1])
    sd_score = np.std(df.iloc[:, 1])
    return mean_score, sd_score

def stats_write_to_csv(filename, data):
    data = data.sort_values('MOS', ascending=False)
    write_to_csv(filename, data)

def write_to_csv(filename, data):
    which_mode = 'a' if os.path.exists(filename) else 'w+'
    data.to_csv(filename, mode=which_mode, header=False)

def make_df(data):
    return pd.DataFrame(data, columns=['speaker', 'MOS', 'sd'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run automated NISQA evaluation')

    parser.add_argument("--fullEval", action='store_true',help="Run all steps of evaluation")
    parser.add_argument("--preprocess", action='store_true',help="Select subtext of German sentences without corrupted characters.")
    parser.add_argument("--generate", action='store_true',help="Generate German samples using Aflorithmic API.")
    parser.add_argument("--apiKey", help="You Aflorithmic API key.")
    parser.add_argument("--predict", action='store_true',help="Run NISQA to predict scores.")
    parser.add_argument("--quality", action='store_true',help="Predict quality scores")
    parser.add_argument("--naturalness", action='store_true',help="Predict naturalness scores") # if neither arg provided, do both
    parser.add_argument("--stats", action='store_true',help="Compute mean & s.d. of NISQA scores for each voice.")
    parser.add_argument("--inRange", action='store_true',  help="Get synthesis files within specified NISQA quality score range and copy files to new folder.")
    parser.add_argument("--min", type=float, help="Minimum value for NISQA range.")
    parser.add_argument("--max", type=float, help="Maximum value for NISQA range.")
    parser.add_argument("--nFiles", type=int, help="Number of files in range to copy to folder.")
    args = parser.parse_args()

    if args.fullEval == True:
        args.preprocess = True
        args.generate = True
        args.predict = True 
        args.stats = True 
        args.inRange = True
    
    if args.preprocess == True:
        to_discard = 'Ã' # characters to exclude
        kept = 1
        discarded = 0
        written=0
        keep_every=4
        with open('eval_config/big_utts.csv', 'r') as inf:
            with open('eval_config/processed_utts.txt', 'w+') as outf:
                for line in csv.reader(inf, delimiter='\t'):
                    utt = (line[0].split(' ', 1)[1])
                    if to_discard not in utt:
                        kept+=1
                        if kept%keep_every == 0: # writing every 4th sentence to file - because there are too many
                            outf.write(f'{utt}\n')
                            written+=1
                    else:
                        discarded+=1
        print(f"There were {kept-1+discarded} sentences in your sentence file.")
        print(f"{kept-1} sentences were kept and {discarded} sentences were discarded because they contained the character(s) {to_discard}.")
        print(f"{written} sentences were written to the final sentence file, because you opted to write every 1/{keep_every} sentences.")
    
    if args.generate == True:
        if args.apiKey == False:
            print("API Key missing: you need to pass your Aflorithmic API Key in order to generate audio.")
        else:
            apiaudio.api_key = args.apiKey
            with open("eval_config/speakers.txt", "r") as o:
                speakers  = o.read().splitlines()
            with open("eval_config/processed_utts.txt", "r") as o:
                utts_list  = o.read().splitlines()
            utts = [{'utt':utt}for utt in utts_list]
            text = """
                <<sectionName::main>> 
                {{utt}}
            """
            script = apiaudio.Script().create(scriptText=text, 
                                                scriptName="German_sample", 
                                                moduleName="German_module", 
                                                projectName="German")
            #Create speech. Choose a voice! https://library.api.audio/voices
            for speaker in speakers:
                if not os.path.exists(os.path.join(f"{os.getcwd()}", speaker)):
                    os.mkdir(os.path.join(f"{os.getcwd()}", speaker))
                for n, utt in enumerate(utts):
                    url = generate_sample(script, speaker, [utt])
                    filename = generate_filepath(speaker, n)
                    download_audio_file(url, filename)
    
    if args.predict == True:
        if args.quality == False and args.naturalness == False:
            args.quality = True 
            args.naturalness = True 
        if args.quality == True:
            pass # add command line flags to shell scrip to handle quality/naturalness modes
        if args.naturalness == True:
            pass
        rc = subprocess.call("./run_predict_batch.sh", shell=True)
    
    factors = {}
    if args.quality == True:
        factors['quality'] = 1
    if args.naturalness == True:
        factors['naturalness'] = 1

    if args.stats == True:
        for factor, col_index in factors.items():
            base_dir = os.path.join(os.getcwd(), f'to_evaluate/results/{factor}')
            speaker_stats = []
            for speaker in os.listdir(base_dir):
                mean_score, sd_score = calculate_mean_score(os.path.join(base_dir, speaker, 'NISQA_results.csv'), col_index)
                speaker_stats.append([speaker, mean_score, sd_score])
            speaker_stats = make_df(speaker_stats)
            stats_write_to_csv(f'mean_NISQA_results_{factor}.csv', speaker_stats)

    if args.inRange is True:
        range_min = args.min if args.min is not None else 4.6
        range_max = args.max if args.max is not None else 4.9
        n_files = args.nFiles if args.nFiles is not None else 15 # default number of files copied per speaker to 15

        csv_filename = f'{n_files}_samples_in_range.csv' 
        wav_dir = f'{os.getcwd()}/to_evaluate/input'
        out_dir = f'{os.getcwd()}/eval_in_range'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if os.path.exists(csv_filename):
            os.remove(csv_filename)

        n_in_range = 0
        for factor, col_index in factors.items():
            base_dir = f"/home/jovyan/NISQA/to_evaluate/results/{factor}" # change this to arg
            for speaker in os.listdir(base_dir):
                if not os.path.exists(os.path.join(out_dir,speaker)):
                    os.mkdir(os.path.join(out_dir,speaker))
                df = return_targets(os.path.join(base_dir, speaker, 'NISQA_results.csv'), col_index, range_min, range_max)
                try:
                    copy_files(df,wav_dir, out_dir, speaker, n_files)
                    n_in_range += inrange_write_to_csv(csv_filename, df,  speaker, n_files)
                except IndexError:
                    print(f"Speaker {speaker} doesn't have {n_files} wav files in the specified range.")
                    continue

