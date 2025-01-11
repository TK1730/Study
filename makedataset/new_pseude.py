import sys
sys.path.append('./')
import shutil
import utils.functions as functions
import utils.config as config
import soundfile as sf

from pathlib import Path
# laudnes
# jvs_nonpara30 + throat
lufs_mix = -38.07440141136585
lufs_jvs = -27.657454012957633
lufs_throat = -54.773324721791454
lufs_nonpara30w = -21.540267870444595

save_path = Path("dataset/jvs_ver2")

dataset_path = Path("dataset/jvs_ver1")
for person in dataset_path.iterdir():
    person = person.joinpath("whisper10", "wav24kHz16bit")
    for data in person.iterdir():
        save_folder = save_path.joinpath(data.parts[2], data.parts[3], data.parts[4])
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        wav = functions.ludness_normalize(data, lufs_average=lufs_throat)
        sf.write(save_folder.joinpath(data.name),wav, samplerate=config.sr, subtype="PCM_16")
