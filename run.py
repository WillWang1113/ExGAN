import os

data_type = ["pv", "wind"]
E = 350

for d in data_type:
    # os.system(f"python DCGAN.py --epochs={E} --data={d}")
    # print("\n DCGAN Done! \n")

    # os.system(f"python DistributionShifting.py --epochs={E} --data={d}")
    # print("\n Distribution Shifted! \n")

    os.system(f"python PrepareData.py --data={d}")
    print("\n Prepared Data! \n")

    os.system(f"python ExGAN.py  --epochs={E} --data={d}")
    print("\n ExGAN Done! \n")

    os.system(f"python ExGANSampling.py   --epochs={E} --data={d}")
    print("\n Sampling Done! \n")