import copy
import os
import pickle
import time
from asyncio import constants
from tabnanny import check

#from museval import metrics
import fast_bss_eval
import IPython.display as ipd
import librosa as lr
import matplotlib.pyplot as plt
import museval
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as scs
import soundfile as sf

import constants
from mixUtils import (createTracksBySambaTypes, getCurrentInst,
                      getInstTypeTrack, writeCompEst, writeEst, writeMix)
from NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.inverseSTFT import inverseSTFT
from NMFtoolbox.NMFD import NMFD
from NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy

################################################################################

# Script variable
#inst_whitelist = ["Tamborim","Pandeiro","Surdo","Shaker"]
inst_whitelist = [   
    "Tamborim",
    "Reco-reco",
    "Caixa",
    "Repique",
    "Cuica",
    "Agogo",
    "Shaker",
    "Tanta",
    "Surdo",
    "Pandeiro"
]

audio_inp_path                      = '/home/tsimoes/tcc/brid/BRID (2018)/Data/Solo Tracks_min'
beat_inp_path                       = '/home/tsimoes/tcc/datasets/BRID/solos/beats'
out_path                            = '/home/tsimoes/tcc/dsp_method/TCC-DSP/output'
save_mix                            = True
save_estimate                       = False
strategy                            = "BRID"
phase_reconstruction_by_reference   = False
max_inst                            = 1
max_inst_2                          = 2 


# create the output directory if it doesn't exist
if not os.path.isdir(out_path):
    os.makedirs(out_path)

if not("tracks_by_samba_types.pkl" in os.listdir()):
    tracks_by_samba_types = createTracksBySambaTypes(audio_inp_path,beat_inp_path,out_path,inst_whitelist)
else:
    tk_by_sb_tp = open(f"./tracks_by_samba_types.pkl", "rb")
    tracks_by_samba_types = pickle.load(tk_by_sb_tp)
    tk_by_sb_tp.close()

if not("samba_type_metrics.pkl" in os.listdir()):
    samba_type_metrics = {}
else:
    sb_metrics = open(f"./samba_type_metrics.pkl", "rb")
    samba_type_metrics = pickle.load(sb_metrics)
    sb_metrics.close()

for samba_type in tracks_by_samba_types.keys():
    # find the mix type number
    if not tracks_by_samba_types[samba_type]["ready"]:
        original_tracks = tracks_by_samba_types[samba_type]["tracks"]  
        all_intruments = []

        for track in original_tracks:
            current_instruments = [getInstTypeTrack(trk) for trk in (track.name).split('+') if not(getInstTypeTrack(trk) in all_intruments)]
            for inst in current_instruments: all_intruments.append(inst)

        num_inst = len(all_intruments)
        # save original inst and next result
        mix = [[],[]]

        mix[0] = copy.deepcopy(original_tracks)
        cont = 2
        inst2 = []
        inst_type = []
        fs = 44100
        #dict de dict das metricas por tipo de samba para cada tipo de instrumento:
        #Ex:
        #samba_type_metrics[0]
        #-> {"SA":{}}
        #samba_type_metrics["SA"]
        #-> {"ag":{"sdr":[],"isr":[],"sir":[],"sar":[],"perm":[]}}
        if not(samba_type in samba_type_metrics.keys()):
            samba_type_metrics.update({samba_type:{2:{}}})
        
        for num_mix_type in range(0,num_inst-1):
            if not((num_mix_type+2) in samba_type_metrics[samba_type].keys()):
                samba_type_metrics[samba_type].update({(num_mix_type+2):{}})
            inst2.clear()
            # to make the copy from the original tracks that will be sum with the current track
            inst_type.clear()
            inst_count_1 = 0
            for track in mix[0]:
                getCurrentInst(track, inst2, inst_type)
                tracks_to_sum = [trk for trk in original_tracks if not((getInstTypeTrack(trk.name) in inst_type))]
                tracks_to_sum = copy.deepcopy(tracks_to_sum)

                if (tracks_to_sum):
                    if not(track.mix):
                        track.setAmpVector((wav.read(track.getPath())[1]))
                        ampTrkVector = track.getAmpVector()
                        ampTrkVector = make_monaural(ampTrkVector)
                        ampTrkVector = pcmInt16ToFloat32Numpy(ampTrkVector)
                        track.setAmpVector(ampTrkVector)
                    inst_count_2 = 0    
                    for track_to_sum in tracks_to_sum:
                        if not(track_to_sum.mix):
                            track_to_sum.setAmpVector((wav.read(track_to_sum.getPath())[1]))
                            ampTrkToSumVector = track_to_sum.getAmpVector()
                            ampTrkToSumVector = make_monaural(ampTrkToSumVector)
                            ampTrkToSumVector = pcmInt16ToFloat32Numpy(ampTrkToSumVector)
                            track_to_sum.setAmpVector(ampTrkToSumVector)
                        #num_mix_type comeca com 0, ai faco a ponderacao das amplitudes
                        mix_track = track*((num_mix_type+1)/((num_mix_type+1)+1)) + track_to_sum*(1/((num_mix_type+1)+1))

                        print(f"processing {mix_track.name}")
                        #######################################SOURCE SEPARATION ALGORITHM###############################
                        #input: 
                        # 1.Mix Track
                        # 2.Output path (optional)
                        # 3.List of N groudtruth paths of the mix track sources 
                        #descomentar a partir daqui
                        ground_truth_tracks = [track for track in original_tracks if track.name in [path.split('/')[-1] for path in mix_track.groudtruth_paths]]
                        ground_truth_tracks = copy.deepcopy(ground_truth_tracks)

                        if(save_mix and inst_count_1 < max_inst and inst_count_2 < max_inst_2):          
                            writeMix(out_path, samba_type, cont, mix_track)
                        # spectral parameters
                        paramSTFT = dict()
                        paramSTFT['blockSize'] = 2048
                        paramSTFT['hopSize'] = 512
                        paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
                        paramSTFT['reconstMirror'] = True
                        paramSTFT['appendFrame'] = True
                        paramSTFT['numSamples'] = len(mix_track.amp_vector)

                        # STFT computation
                        X, A, P = forwardSTFT(mix_track.getAmpVector(), paramSTFT)

                        # get dimensions and time and freq resolutions
                        numBins, numFrames = X.shape
                        deltaT = paramSTFT['hopSize'] / fs
                        deltaF = fs / paramSTFT['blockSize']

                        initW = list()
                        inst_components = [0] #guarda o index dos componentes por instrumento presente na mistura
                        if (strategy  == 'BRID'):
                            count = 0
                            for inst in mix_track.inst:
                                template_file = open(f"./templates/{constants.brid_nmfd_tamplates_names[inst]}",'rb')
                                dictW = pickle.load(template_file)
                                template_file.close()
                                for component in dictW.keys():
                                    initW.append(dictW[component])
                                    count+=1
                                inst_components.append(count)
                        else:
                            # generate initial guess for templates
                            paramTemplates = dict()
                            paramTemplates['deltaF'] = deltaF
                            paramTemplates['numComp'] = numComp
                            paramTemplates['numBins'] = numBins
                            paramTemplates['numTemplateFrames'] = numTemplateFrames
                            initW = initTemplates(paramTemplates) # antes estava como drums

                        # set common parameters
                        numComp = len(initW)
                        #numComp = 5
                        numIter = 30
                        numTemplateFrames = 8

                        # generate initial activations
                        paramActivations = dict()
                        paramActivations['numComp'] = numComp
                        paramActivations['numFrames'] = numFrames

                        initH = initActivations(paramActivations,'uniform')

                        # NMFD parameters
                        paramNMFD = dict()
                        paramNMFD['numComp'] = numComp
                        paramNMFD['numFrames'] = numFrames
                        paramNMFD['numIter'] = numIter
                        paramNMFD['numTemplateFrames'] = numTemplateFrames
                        paramNMFD['initW'] = initW
                        paramNMFD['initH'] = initH

                        # NMFD core method
                        paramConstr = dict()
                        paramConstr['funcPointerPostProcess'] = semiFixedComponentConstraintsNMF
                        paramConstr['adaptDegree'] = 1 #0 = fixed, 1 = semi-fixed, 2 = adaptive
                        paramConstr['adaptTiming'] = 4 #http://dafx14.fau.de/papers/dafx14_christian_dittmar_real_time_transcription_a.pdf

                        nmfdW, nmfdH, nmfdV, divKL, _ = NMFD(A, paramNMFD, paramConstr)

                        # alpha-Wiener filtering
                        nmfdA, _ = alphaWienerFilter(A, nmfdV, 1.0)

                        inst_estimates = []
                        # resynthesize results of NMF with soft constraints and score information
                        if phase_reconstruction_by_reference:
                            # identificar as estimativas:
                            for ic in range(0,len(inst_components)-1):
                                phase_template_file = open(f"./templates_phase/{constants.brid_nmfd_phase_tamplates_names[mix_track.inst[ic]]}",'rb')
                                c_phase = pickle.load(phase_template_file)
                                phase_template_file.close()
                                sum_nmfd = nmfdA[inst_components[ic]]
                                for isum in range(inst_components[ic]+1,inst_components[ic+1]):
                                    sum_nmfd += nmfdA[isum]
                                if sum_nmfd[1].size > c_phase[1].size:
                                    c_phase = np.pad(c_phase,((0,0),(0,(sum_nmfd[1].size - c_phase[1].size))),"wrap")
                                else:
                                    c_phase = c_phase[:, 0:sum_nmfd[1].size]
                                Y = sum_nmfd * np.exp(1j * c_phase)
                                y, _ = inverseSTFT(Y, paramSTFT)
                                inst_estimates.append(y)
                                                                  
                        else:
                            audio_estimates = []
                            for k in range(numComp):
                                Y = nmfdA[k] * np.exp(1j * P)
                                y, _ = inverseSTFT(Y, paramSTFT)
    
                                audio_estimates.append(y)
                                if (save_mix and inst_count_1 < max_inst and inst_count_2 < max_inst_2):
                                    writeCompEst(out_path, samba_type, cont, mix_track, y, f"{k}")
                                #wav.write(filename=f"{k}componentes-sem-temp.wav", rate=fs, data=y)
                                 
                            # identificar as estimativas:
                            # o template inicial 
                            for ic2 in range(0,len(inst_components)-1):
                                sum_comp = audio_estimates[inst_components[ic2]]
                                for isum in range(inst_components[ic2]+1,inst_components[ic2+1]):
                                    sum_comp += audio_estimates[isum]
                                    #print("somou!")
                                inst_estimates.append(sum_comp)
                                if (save_mix and inst_count_1 < max_inst and inst_count_2 < max_inst_2):
                                    writeEst(out_path, samba_type, cont, mix_track, sum_comp, ic2)

                        zipao = zip(inst_estimates,ground_truth_tracks)
                        inst_estimates_eval = []
                        ground_truth_tracks_eval = []
                        for estimate,ground_truth in zipao:
                            # Ao misturar teve ponderamento das amplitudes, para voltar é necessário desfazer esse ponderamento
                            estimate = (estimate.T[0])/(1/((num_mix_type+1)+1))

                            #inicializando o dicionario de metricas para os instrumentos
                            if not(getInstTypeTrack(ground_truth.name) in samba_type_metrics[samba_type][num_mix_type+2].keys()):
                                samba_type_metrics[samba_type][num_mix_type+2].update({getInstTypeTrack(ground_truth.name):{"sdr":[],"isr":[],"sir":[],"sar":[],"perm":[]}})
                            # Cortando no primeiro beat
                            ground_truth.setAmpVector((wav.read(ground_truth.getPath())[1][int(ground_truth.fs*ground_truth.first_beat):]))
                            ref = make_monaural((ground_truth.getAmpVector()))
                            ref = pcmInt16ToFloat32Numpy(ref)

                            inst_estimates_eval.append(estimate)
                            ground_truth_tracks_eval.append(ref[:estimate.size])
                        (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(ground_truth_tracks_eval,inst_estimates_eval,window=np.inf)
                        #(sdr, sir, sar, perm) = fast_bss_eval.bss_eval_sources(ref[:estimate.size],estimate)
                        #(sdr, isr, sir, sar) = museval.evaluate(ref[:estimate.size],estimate,win=np.inf)
                        for i in range(0,len(ground_truth_tracks)):

                            samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth_tracks[i].name)]["sdr"].append(sdr[i][0])
                            samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth_tracks[i].name)]["isr"].append(isr[i][0])
                            samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth_tracks[i].name)]["sir"].append(sir[i][0])
                            samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth_tracks[i].name)]["sar"].append(sar[i][0])
                            samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth_tracks[i].name)]["perm"].append(perm[i][0])
                            #samba_type_metrics[samba_type][num_mix_type+2][getInstTypeTrack(ground_truth.name)]["perm"].append(perm[0][0])
                            #print(f"ref: {ground_truth_tracks[iMax].name}")
                            #i+=1
                            #wav.write(filename=f"{ground_truth.name}{mix_track.name}-estimate.wav", rate=fs, data=estimate)

                        #rever documentação das metricas
                        ##################################################################################################
                        ##mix_track.clearAmpVector()
                        if not(constants.brid_inst[inst_whitelist[-1]] in mix_track.inst):
                            mix_track.setPath(f"{out_path}/{samba_type}/{cont}/{mix_track.name}")
                            mix[1].append(mix_track)
                        del mix_track
                        #del audio_estimates
                        inst_count_2 += 1
                del tracks_to_sum
                #cont_track += 1
                inst_count_1 += 1
            mix[0].clear()
            mix[0] = [t for t in mix[1]]         
            mix[1].clear()      
            cont+=1

        del all_intruments
        del original_tracks
        del mix
        del inst2
        del inst_type

        ##########
        #for inst in samba_type_metrics[samba_type]:
        #    av_sdr = np.average(samba_type_metrics[samba_type][inst]["sdr"])
        #    av_isr = np.average(samba_type_metrics[samba_type][inst]["isr"])
        #    av_sir = np.average(samba_type_metrics[samba_type][inst]["sir"])
        #    av_sar = np.average(samba_type_metrics[samba_type][inst]["sar"])
        #    av_perm = np.average(samba_type_metrics[samba_type][inst]["perm"])
        #    print(f"{inst}: {samba_type}\n\tAverage SDR: {av_sdr}\n\tAverage ISR: {av_isr}\n\tAverage SIR: {av_sir}\n\tAverage SAR: {av_sar}\n\tAverage PERM: {av_perm}")

        # Salvando objeto das metricas em arquivo
        sb_metrics = open(f"./samba_type_metrics.pkl", "wb")
        pickle.dump(samba_type_metrics, sb_metrics)
        sb_metrics.close()

        tk_by_sb_tp = open(f"./tracks_by_samba_types.pkl", "wb")
        pickle.dump(tracks_by_samba_types, tk_by_sb_tp)
        tk_by_sb_tp.close()
        
        tracks_by_samba_types[samba_type]["ready"] = True
