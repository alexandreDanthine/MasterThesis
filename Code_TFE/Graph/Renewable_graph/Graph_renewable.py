from Renewable_data import read_from_excel_wind, read_from_excel_sun, read_from_excel_demand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


data_w = read_from_excel_wind('data/WindForecast_2015-12-31_2016-01-31.xlsx')
data_s = read_from_excel_sun('data/SolarForecast_2015-12-31_2016-01-31.xlsx')
data_d = read_from_excel_demand('data/Load_Day_Ahead_2016_2017.xlsx')



hour = np.linspace(0,24,96)

plt.figure()
for i in range(31):
	plt.plot(hour,data_s.day_ahead_sun[i])
plt.xlabel("Time [hours]")
plt.ylabel("Power generated [MW]")
plt.xlim(0,24)
plt.savefig("Solar_prediction.pdf")


plt.figure()
for i in range(31):
	plt.plot(hour,data_w.day_ahead_wind[i])
plt.xlabel("Time [hours]")
plt.ylabel("Power generated [MW]")
plt.xlim(0,24)
plt.savefig("Wind_prediction.pdf")



renewable_prod_day_ahead = np.zeros((31,96))
renewable_prod_imbalance = np.zeros((31,96))


for i in range(31):
	for j in range(96):
		renewable_prod_day_ahead[i,j] = data_s.day_ahead_sun[i][j] + data_w.day_ahead_wind[i][j]
		renewable_prod_imbalance[i,j] = (data_s.measured_sun[i][j] + data_w.measured_wind[i][j]) - (data_s.day_ahead_sun[i][j] + data_w.day_ahead_wind[i][j])


#print renewable_prod_day_ahead[0,:]



plt.figure()
for i in range(31):
	plt.plot(hour,renewable_prod_day_ahead[i,:])
plt.xlabel("Time [hours]")
plt.ylabel("Power generated [MW]")
plt.xlim(0,24)
plt.savefig("Renewable_prediction.pdf")






plt.figure()

plt.plot(hour,renewable_prod_imbalance[14,:])
plt.plot(np.zeros(25), color = 'k',linewidth=2.0)
plt.xlabel("Time [hours]")
plt.ylabel("Imbalance power [MW]")
plt.xlim(0,24)
plt.savefig("Renewable_imbalance.pdf")



renewable_std_imbalance = np.zeros(96)

for i in range(96):
	renewable_std_imbalance[i] = np.std(renewable_prod_imbalance[:,i])


plt.figure()
plt.plot(hour,renewable_std_imbalance,linewidth=2.0)
plt.xlabel("Time [hours]")
plt.ylabel("Standard deviation of imbalance power [MW]")
plt.xlim(0,24)
plt.savefig("Renewable_standard_deviation.pdf")


#print(data_d.volume_demand[1])

plt.figure()
for i in range(31):
	plt.plot(hour,data_d.volume_demand[i])
plt.xlabel("Time [hours]")
plt.ylabel("Demand [MW]")
plt.xlim(0,24)
plt.savefig("Demand.pdf")




plt.show()


