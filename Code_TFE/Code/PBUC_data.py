from __future__ import division
import xlrd
from Data import Data


class SizingData(Data):

	def __init__(self, *args, **kwargs):
		super(SizingData, self).__init__(*args, **kwargs)

	def assert_data(self):
		pass

def read_from_excel_3(file_path,case_id=1):
	
	s_reserve  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('Not_specified')

	data_3 = SizingData(
						up_activated = [[float(s_reserve.cell_value(10+(i*95)+j+4*i,5)) for j in range(96)]for i in range(31)],
						down_activated = [[float(s_reserve.cell_value(10+(i*95)+j+4*i,9)) for j in range(96)]for i in range(31)]
						)


	return data_3

def read_from_excel_2(file_path,case_id=1):
	
	s_price  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('prices_2')
	s_demand  = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('volumes_2')
	s_reserve = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('AGC_price')
	h = 24

	data_2 = SizingData(hours = h,
						price = [[float(s_price.cell_value(j+1, i+1)) for i in range(h)]for j in range(31+25)],
						forecast_price = [float(s_price.cell_value(21, i+1)) for i in range(h)],
						actual_price = [float(s_price.cell_value(2, i+1)) for i in range(h)],
						#volume_demand = [float(s_demand.cell_value(1, i+1)) for i in range(h)],
						AGC_price = float(s_reserve.cell_value(1,0)),
						AGC_capacity = float(s_reserve.cell_value(1,1))
						)


	return data_2

def read_from_excel_1(file_path,case_id=1):

	s_general = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('general')
	s_generator = xlrd.open_workbook(file_path, on_demand=True).sheet_by_name('generator')
	np=int(s_general.cell_value(case_id, 1))
	ng = 11
	#11 normalement 
	data_1 = SizingData(nperiods=np,
						  ngenerators = ng,
						  periodDuration=float(s_general.cell_value(case_id, 2)),
						  powerMax = [float(s_generator.cell_value(i+1,0)) for i in range(ng)],
						  countInit = [float(s_generator.cell_value(i+1,1)) for i in range(ng)],
						  onOffInit = [float(s_generator.cell_value(i+1,2)) for i in range(ng)],
						  fixedCost = [float(s_generator.cell_value(i+1,3)) for i in range(ng)],
						  rampUpLim = [float(s_generator.cell_value(i+1,4)) for i in range(ng)],
						  rampDownLim = [float(s_generator.cell_value(i+1,5)) for i in range(ng)],
						  minUpTime = [float(s_generator.cell_value(i+1,6)) for i in range(ng)],
						  minDownTime = [float(s_generator.cell_value(i+1,7)) for i in range(ng)],
						  powerMin = [float(s_generator.cell_value(i+1,8)) for i in range(ng)],
						  powerInit = [float(s_generator.cell_value(i+1,9)) for i in range(ng)],
						  powerCost = [[float(s_generator.cell_value(i+1,10+j)) for j in range(3)]for i in range(ng) ],
						  rangeCost = [[float(s_generator.cell_value(i+1,13+j)) for j in range(3)] for i in range(ng)],
						  startUpCost = [float(s_generator.cell_value(i+1,16)) for i in range(ng)]
						  )


	return data_1
	




if __name__ == '__main__':


	A = read_from_excel_1('data/Personal_data.xlsx')
	#B = read_from_excel_2('data/spotmarket_data_2016.xls')
	#C = read_from_excel_3('data/Reserve_2016.xlsx')
	print(A)
	#print(B)
	#print(C)

	#powerCost = np.array(np.asarray([[model.powerGen[i,j].value for j in model.Generators] for i in model.Periods]))






















