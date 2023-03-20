import os

data_type = "wind"
epochs = 50

os.system(f"python DCGAN.py --epochs={epochs} --data={data_type}")
print("\n DCGAN Done! \n")

os.system(
    f"python DistributionShifting.py --epochs={epochs} --data={data_type}")
print("\n Distribution Shifted! \n")

os.system(f"python PrepareData.py --data={data_type}")
print("\n Prepared Data! \n")

os.system(f"python ExGAN.py  --epochs={epochs} --data={data_type}")
print("\n ExGAN Done! \n")

os.system(f"python ExGANSampling.py   --epochs={epochs} --data={data_type}")
print("\n Sampling Done! \n")