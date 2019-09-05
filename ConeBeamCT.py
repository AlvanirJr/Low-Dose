import astra
import numpy as np



class ConeBeamCT():

    def __init__(self, volume):

        self.__volume = volume



    def run_new_scan(self, n_projections):
        x, y, z = self.__volume.shape
        print(x,y,z)
        vol_geom = astra.create_vol_geom(y, z, x)

        angles = np.linspace(0, np.pi, n_projections,False)
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 512, 512, angles,300000,300)
        proj_id, proj_data = astra.create_sino3d_gpu(self.__volume, proj_geom, vol_geom)

        rec_id = astra.data3d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

        return rec



if __name__ == "__main__":

    import pylab
    from DataInterface import DataInterface

    src = "/home/andrei/Área de Trabalho/Pesquisa/DATASET-256/"
    dataset = DataInterface(src)

    vol = dataset.get_tomo_volume(90)
    ct = ConeBeamCT(vol)

    rec = ct.run_new_scan(15)


    pylab.gray()
    pylab.figure(1)
    pylab.imshow(vol[:,:,128])

    pylab.figure(2)
    pylab.imshow(rec[:,:,128])
    pylab.show()


