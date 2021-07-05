import spencer

f = "slide_3_measurements.csv"
expName = "8types"
outFolder = f"out/spencer/{expName}"

cells = spencer.loadMarkers(f)

types = np.loadtxt(f"{outFolder}/{f[:-4]}types.csv")