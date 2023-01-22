from Track import Track 
import soundfile as sf
import os
import scipy.io.wavfile as wav

# create the list of samba types with the track's names in it
def createTracksBySambaTypes(audio_inp_path,beat_inp_path,out_path, inst_whitelist = ["Agogo","Caixa","Cuica","Pandeiro","Reco-reco","Repique","Shaker","Surdo","Tamborim","Tanta"]):
    instrumentalists = os.listdir(audio_inp_path)
    tracks_by_samba_types = {}

    for instrumentalist_index in range(0, len(instrumentalists)):
        instruments_path = os.path.join(
            audio_inp_path, instrumentalists[instrumentalist_index])
        instruments = os.listdir(instruments_path)
        beats = os.listdir(beat_inp_path)
        instruments = [inst for inst in instruments if inst in inst_whitelist]
        for instrument_index in range(0, len(instruments)):
            tracks_path = os.path.join(
                instruments_path, instruments[instrument_index])
            tracks = os.listdir(tracks_path)
            for track in tracks:
                if (f"{track[:-3]}beats" in beats):
                    current_samba_type = getSambaType(track)
                    if not(current_samba_type in tracks_by_samba_types.keys()):
                        tracks_by_samba_types.update({current_samba_type: {"tracks":list(),"ready":False}})
                    beats_fd = open(os.path.join(beat_inp_path,f"{track[:-3]}beats"))
                    beats_points = beats_fd.readlines()
                    beats_points = [d[0] for d in [([c.split("\t") for c in b.split("\n")]) for b in beats_points]]
                    find = False
                    count = 0
                    
                    # sincronizando pelo primeiro beat
                    while not find and count < len(beats_points):
                        if int(beats_points[count][1]) == 1:
                            first_beat = beats_points[count][0]
                            find = True
                        count += 1
                    # Convencionar um numero de amostras antes do primeiro beat para adotar uma zona de seguranca (silencio), não perder informação    
                    safe_zone = 0.01
                    m_first_beat = float(first_beat)-safe_zone
                    if m_first_beat < 0:
                        m_first_beat = 0
                    trk = Track(track, m_first_beat)

                    if not os.path.isdir(f"{out_path}/{current_samba_type}"):
                        os.makedirs(f"{out_path}/{current_samba_type}/1")
                    #print(f"cp {trk.getPath()} {out_path}/{current_samba_type}/1")
                    os.system(f"cp '{tracks_path}/{track}' '{out_path}/{current_samba_type}/1'")
                    trk.setPath(f"{out_path}/{current_samba_type}/1/{track}")
                    tracks_by_samba_types[current_samba_type]["tracks"].append(trk)
                    beats_fd.close()
                else:
                    pass
                    #print(f"{track[:-3]}beats not found!")
    return tracks_by_samba_types

def getSambaType(track):
    return ((track.split('-'))[-1]).split('.')[0]
def getInstTrack(track):
    return ((track.split(' ')[1]).split('.')[0])
def getInstTypeTrack(inst_track):
    if ('-' in inst_track):
        return (inst_track.split('-')[1])[0:2]
    return (inst_track.split(' ')[1])[2:4]
def getNumTrack(track):
    return ((track.split(' ')[0])[1:-1])

# Create the mix
def getCurrentInst(track, inst, inst_type):
    current_type_instruments = [getInstTypeTrack(trk) for trk in (track.name).split('+')]
    current_instruments = [getNumTrack(trk) for trk in (track.name).split('+')]
    for i in range(0,len(current_type_instruments)):
    # Se tiver o tipo, mas não estiver na lista de instrumentos, retire os tipos de instrumentos a direita para ser preenchido nas proximas iteracoes
        if (current_type_instruments[i] in inst_type and not(current_instruments[i] in inst)):
            for s_type in inst_type[inst_type.index(current_type_instruments[i])+1:]: inst_type.remove(s_type)
            inst.append(current_instruments[i])
        elif not(current_type_instruments[i] in inst_type):
            if not(current_instruments[i] in inst):    
                inst.append(current_instruments[i])
            inst_type.append(current_type_instruments[i])
        else:
            if i == (len(current_type_instruments)-1):
                for s_type in inst_type[inst_type.index(current_type_instruments[i])+1:]: inst_type.remove(s_type)
    del current_type_instruments
    del current_instruments

def writeMix(out_path, sambaType, cont, mix_track, label="mix"):
    if not os.path.isdir(f"{out_path}/{sambaType}/{cont}"):
        os.makedirs(f"{out_path}/{sambaType}/{cont}")
    #sf.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name}", mix_track.amp_vector, 44100)
    wav.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name[:-4] + '-' + label + '.wav'}", rate=44100, data=mix_track.amp_vector)

def writeEst(out_path, sambaType, cont, mix_track, est, ic2):
    if not os.path.isdir(f"{out_path}/{sambaType}/{cont}"):
        os.makedirs(f"{out_path}/{sambaType}/{cont}")
    #sf.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name}", mix_track.amp_vector, 44100)
    wav.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name[:-4] + '-' + mix_track.inst[ic2] + '.wav'}", rate=44100, data=est)
def writeCompEst(out_path, sambaType, cont, mix_track, est, label):
    if not os.path.isdir(f"{out_path}/{sambaType}/{cont}"):
        os.makedirs(f"{out_path}/{sambaType}/{cont}")
    #sf.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name}", mix_track.amp_vector, 44100)
    wav.write(f"{out_path}/{sambaType}/{cont}/{mix_track.name[:-4] + '-' + label + '.wav'}", rate=44100, data=est)

