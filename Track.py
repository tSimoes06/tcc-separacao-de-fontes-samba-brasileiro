import mixUtils as mu
import numpy as np
class Track:
    
    def __init__(self, name, first_beat=0.0, fs=44100) -> None:
        self.name = name
        self.first_beat = first_beat
        self.track_path = None
        self.groudtruth_paths = []
        self.inst = []
        self.fs = fs
        self.amp_vector = None
        self.mix = False
    # adding two objects
    def __add__(self, o):
        # se não tiver somado
        if not(self.groudtruth_paths):
            self.addGTPath(self.track_path)
            self.addGTPath(o.track_path)
        else:
            self.addGTPath(o.track_path)
        if not(self.inst):
            self.inst.append(mu.getInstTypeTrack(self.name))
            self.inst.append(mu.getInstTypeTrack(o.name))
        else:
            if not(mu.getInstTypeTrack(o.name) in self.inst):
                self.inst.append(mu.getInstTypeTrack(o.name))
        #alinhando os vetores de acordo com os primeiros beats
        if (o.first_beat):
            o.amp_vector = o.amp_vector[int(self.fs*o.first_beat):]
        if (self.first_beat):
            self.amp_vector = self.amp_vector[int(self.fs*self.first_beat):]
        #cortando o maior para ficar com o tamanho do menor e somar
        if ((self.amp_vector).size <= (o.amp_vector).size):
            mix_amp_vector =  (self.amp_vector + o.amp_vector[0:(self.amp_vector).size])
        else:
            mix_amp_vector = (self.amp_vector[0:(o.amp_vector).size] + o.amp_vector)
        if ('+' in self.name):
            mix_name = f"{self.name[:-4]}+{(o.name).split('-')[0]+(o.name).split('-')[1]}.wav"
        else:
            mix_name = f"{(self.name).split('-')[0]+(self.name).split('-')[1]}+{(o.name).split('-')[0]+(o.name).split('-')[1]}.wav"
        aux = Track (mix_name)
        aux.setAmpVector(mix_amp_vector)
        aux.mix = True
        for path in self.groudtruth_paths: aux.addGTPath(path)
        for inst in self.inst: aux.inst.append(inst)
        return aux
        #return Track(mix_name)
    def __mul__(self, num):
        aux = Track(self.name,self.first_beat)
        aux.setAmpVector(self.amp_vector*num)
        aux.setPath(self.track_path)
        aux.mix = self.mix
        for path in self.groudtruth_paths: aux.addGTPath(path)
        for inst in self.inst: aux.inst.append(inst)
        return aux
        
    __rmul__ = __mul__   

    def getAmpVector(self):
        return self.amp_vector
    # para não ficar guardando em memoria antes sem necessidade, so calcula quando for escrever o wav e  ja deixo alinhado no primeiro beat
    
    def setAmpVector(self, amp_vector):
        # Alinho no momento de definir
        self.amp_vector = amp_vector

    def setPath(self, track_path):
        self.track_path = track_path

    def getPath(self):
        return self.track_path 

    def clearAmpVector(self):
        self.amp_vector = None

    def addGTPath(self,path):
        self.groudtruth_paths.append(path)

    
