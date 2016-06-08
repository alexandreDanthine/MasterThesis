import xlrd
from Data import Data

class SizingData(Data):

	def __init__(self, *args, **kwargs):
		super(SizingData, self).__init__(*args, **kwargs)

	def assert_data(self):
		pass


def read_from_excel_wind(file_path,case_id=1):

	s_wind = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('forecast')


	data = SizingData(
						day_ahead_wind = [[float(s_wind.cell_value(4+(i*96)+j,2)) for j in range(96)] for i in range(31)],
						measured_wind = [[float(s_wind.cell_value(4+(i*96)+j,4)) for j in range(96)]for i in range(31)],
						  )


	return data

def read_from_excel_sun(file_path,case_id=1):

	s_sun = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('SolarForecasts')


	data = SizingData(
						day_ahead_sun= [[float(s_sun.cell_value(4+(i*96)+j,1)) for j in range(96)]for i in range(31)],
						measured_sun = [[float(s_sun.cell_value(4+(i*96)+j,4)) for j in range(96)]for i in range(31)],
						  )


	return data

def read_from_excel_demand(file_path,case_id=1):
	
	s_demand  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('Sheet1')

	data = SizingData(
						volume_demand = [[float(s_demand.cell_value(8+(i*95)+j+4*i,2)) for j in range(96)]for i in range(31)]
						)

	return data



if __name__ == '__main__':


	print("Data read..")

	data_d = read_from_excel_demand('data/Load_Day_Ahead_2016_2017.xlsx')

	print(data_d)






















