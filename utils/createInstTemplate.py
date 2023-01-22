# Objetivo: criar templates da NMF para todos os instrumentos da BRID
# Pegar um arquivo de cada instrumento, inicialmente seria agogo, calcular
# a STFT do arquivo todo e com a marcação no tempo separar os blocos de cada
# articulação e fazer a media dessas articulações. Colocar os blocos no mesmo
# tamanho

import numpy as np
import scipy.io.wavfile as wav
import pickle
import os
from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy

hopsize                                                = 512
blocksize                                              = 2048
template_frames                                        = 8                                                    

template_inputs_wav_path                               = "./template_inputs_wav/"
template_inputs_onsets_path                            = "./template_inputs_onsets/"
template_output_path                                   = "./templates/"
template_phase_output_path                             = "./templates_phase/"
template_inputs_wav                                    = os.listdir(template_inputs_wav_path)
template_inputs_onsets                                 = os.listdir(template_inputs_onsets_path)

for index in range(0,len(template_inputs_wav)):
    file = template_inputs_wav[index][:-4]
    onsets = np.loadtxt(os.path.join(template_inputs_onsets_path, template_inputs_onsets[index]))

    # carrega o sinal de audio
    fs, x = wav.read(os.path.join(template_inputs_wav_path, template_inputs_wav[index]))

    # make monaural if necessary
    x = make_monaural(x)
    x = pcmInt16ToFloat32Numpy(x)

    # parametros espectrais
    paramSTFT                                           = dict()
    paramSTFT['blockSize']                              = blocksize
    paramSTFT['hopSize']                                = hopsize
    paramSTFT['winFunc']                                = np.hanning(paramSTFT['blockSize'])
    paramSTFT['reconstMirror']                          = True
    paramSTFT['appendFrame']                            = True
    paramSTFT['numSamples']                             = len(x)

    # Computa a STFT
    _, A, P = forwardSTFT(x, paramSTFT)

    components = {}
    # energia
    E = np.sum(A**2,axis=0)
    for onset_index in range(0, len(onsets)):
        # se ainda não tiver aquela articulação no dicionario de components cria uma entrada para ela
        if not(onsets[onset_index][1] in components.keys()):
            components.update({onsets[onset_index][1]: list()})
        init_frame = round(onsets[onset_index][0]*fs/hopsize)
        # o ultimo frame para o ultimo onset precisa ser o final do arquivo
        if onset_index != (len(onsets)-1):
            final_frame = round(onsets[onset_index + 1][0]*fs/hopsize)
        else:
            final_frame = round(len(x)/hopsize)
        # checando se a duração do evento passa o tamanho do template desejado que no caso é de 8 frames, se for tem que cortar em 8
        # calcular a energia de cada frame e pegar o frame
        if (final_frame - init_frame >= template_frames):
            energy_event = E[init_frame:init_frame+(template_frames)]
            max_event_frame = np.argmax(energy_event, axis=None)

            # checar se o onset está alinhado com a energia do evento
            if (max_event_frame):
                init_frame += max_event_frame
                # final_frame+=max_event_frame
            # margem de seguranca de 1 frame para garantir que o onset esteja no evento e o max estar no frame 1
            safe_init_event_frame = init_frame - 1
            if safe_init_event_frame < 0:
                safe_init_event_frame = 0    
            inst_event_mod_spec = A[:, safe_init_event_frame:safe_init_event_frame+(template_frames)]
            components[onsets[onset_index][1]].append(inst_event_mod_spec)
    # Soma todos os templates e gera uma media para cada articulação
    average_components = {}
 
    for component in components.keys():
        template_sum = components[component][0]
        num_templates = len(components[component])
        for template_i in range(1, num_templates):
            template_sum = template_sum + components[component][template_i]
        average_template = template_sum/num_templates
        average_components[component] = average_template
        index += 1
    
    template_phase_file = open(os.path.join(template_phase_output_path, f"{file}-phase-template.pkl"), "wb")
    pickle.dump(P, template_phase_file)
    template_phase_file.close()

    template_file = open(os.path.join(template_output_path, f"{file}-template.pkl"), "wb")
    pickle.dump(average_components, template_file)
    template_file.close()
