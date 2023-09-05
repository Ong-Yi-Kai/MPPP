from src.MPPP import *

directory_output = './data/output/'
IMG_paths = ["./data/input/ZR0_0792_0737263387_769RAD_N0390926ZCAM01023_034050A01.IMG"] * 100
sol = '792'
suf = 'refs_'+str(sol).zfill(3)+'_zcam'

if __name__ == "__main__":
    start_normal = time.time()
    image_list_process(IMG_paths[:], directory_output, suf, find_offsets_mode=1)
    end_normal = time.time()


    start_pooled = time.time()
    image_list_process_pooled(IMG_paths[:], directory_output, suf, find_offsets_mode=1)
    end_pooled = time.time()

    print(f"total time taken for-looped: {end_normal - start_normal:.4f}s")
    print(f"total time taken pooled: {end_pooled - start_pooled:.4f}s")