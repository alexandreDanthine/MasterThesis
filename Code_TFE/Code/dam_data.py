
import xlrd
from Data import Data

class SizingData(Data):

	def __init__(self, *args, **kwargs):
		super(SizingData, self).__init__(*args, **kwargs)

	def assert_data(self):
		pass

def read_from_excel(file_path,case_id=1):

	s_bids = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('offers')
	nbids = int(s_bids.cell_value(0,6))
	s_rank = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('rank')

	ngenerators = 11

	#s_demand = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('demand')

	data = SizingData(
						nperiods=96,
						ngenerators = ngenerators,
						nbids = int(s_bids.cell_value(0,6)),
						day = int(s_bids.cell_value(0,7)),
						QuantityBid = [float(s_bids.cell_value(i,0)) for i in range(nbids)],
						PriceBid = [float(s_bids.cell_value(i,1)) for i in range(nbids)],
						TypeBid = [float(s_bids.cell_value(i,2)) for i in range(nbids)],
						TimestepBid = [int(s_bids.cell_value(i,3)) for i in range(nbids)],
						GeneratorBid = [int(s_bids.cell_value(i,4)) for i in range(nbids)],
						Ranking  = [int(s_rank.cell_value(0,i)) for i in range(ngenerators)],
						PowerMax  = [float(s_rank.cell_value(1,i)) for i in range(ngenerators)],
						PriceMax = [float(s_rank.cell_value(2,i)) for i in range(ngenerators)],
						  )


	return data

def read_from_excel_wind(file_path,case_id=1):

	s_wind = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('forecast')


	data = SizingData(
						day_ahead_wind = [[float(s_wind.cell_value(4+(i*95)+j,2)) for j in range(96)] for i in range(31)],
						measured_wind = [[float(s_wind.cell_value(4+(i*95)+j,4)) for j in range(96)]for i in range(31)],
						  )


	return data

def read_from_excel_sun(file_path,case_id=1):

	s_sun = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('SolarForecasts')


	data = SizingData(
						day_ahead_sun= [[float(s_sun.cell_value(4+(i*95)+j,1)) for j in range(96)]for i in range(31)],
						measured_sun = [[float(s_sun.cell_value(4+(i*95)+j,4)) for j in range(96)]for i in range(31)],
						  )


	return data

def read_from_excel_demand(file_path,case_id=1):

	s_demand  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('Sheet1')

	data = SizingData(
						volume_demand = [[float(s_demand.cell_value(8+(i*95)+j+4*i,2)) for j in range(96)]for i in range(31)]
						)

	return data

def read_from_excel_price(file_path,case_id=1):
	
	s_price  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('prices_2')
	h = 24
	data = SizingData(	
						hours = h,
						price = [[float(s_price.cell_value(j+1, i+1)) for i in range(h)]for j in range(31+25)]
						)

	return data

def read_from_excel_imbalance(file_path,case_id=1):
	
	s_imbalance  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('Balancing Prices 2012')
	data = SizingData(	
						up_price = [[float(s_imbalance.cell_value(i+j+2, 9)) for i in range(96)]for j in range(31)],
						down_price = [[float(s_imbalance.cell_value(i+j+2, 10)) for i in range(96)]for j in range(31)],
						)

	return data



if __name__ == '__main__':


	#A = read_from_excel('data/Bids_created.xlsx')
	#print(A)

	#WIND = read_from_excel_wind('data/WindForecast_2015-12-31_2016-01-31.xlsx')
	#print(WIND)
	#SUN = read_from_excel_sun('data/SolarForecast_2015-12-31_2016-01-31.xlsx')
	#print(SUN)

	IMB = read_from_excel_imbalance('data/Imbalance-2016-01.xls')
	print(IMB)





