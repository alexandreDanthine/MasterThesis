## @package dam
# @author Sebastien MATHIEU

from __future__ import division
from pyomo.environ import *
from pyomo.opt import ProblemFormat
from dam_data import read_from_excel, read_from_excel_wind, read_from_excel_sun, read_from_excel_demand, read_from_excel_price, read_from_excel_imbalance
import numpy as np
import math
import unittest
import openpyxl
import xlsxwriter
from pyomo.opt import SolverFactory





data_f = read_from_excel('data/Bids_created.xlsx')
data_w = read_from_excel_wind('data/WindForecast_2015-12-31_2016-01-31.xlsx')
data_s = read_from_excel_sun('data/SolarForecast_2015-12-31_2016-01-31.xlsx')
data_d = read_from_excel_demand('data/Load_Day_Ahead_2016_2017.xlsx')
data_p = read_from_excel_price('data/spotmarket_data_2016.xls')
data_i = read_from_excel_imbalance('data/Imbalance-2016-01.xls')

#OPTIONS
## Verbose mode.
VERBOSE = True

## Debug mode.
DEBUG = True

## Solver.
SOLVER = SolverFactory('gurobi')
#SOLVER.options['mipgap'] = 0.002
if SOLVER is None:
    raise Exception('Unable to instanciate the solver.')

## Numerical accuracy.
EPS = 1e-4


## Day-head energy market agent.
class DAM:
    ## Constructor
    # @param priceCap Tuple with the minimum and maximum price.
    # @param connections Dictionary where the keys are 2-tuple of location and the values the capacity.
    def __init__(self, priceCap=(-3000, 3000), connections={}):
        self.orders = {}
        self.priceCap = priceCap
        self.connections = connections

    ## Set the connections between the localisations.
    # @param connections Dictionary where the keys are 2-tuple of location and the values the capacity.
    def setConnections(self, connections={}):
        self.connections = connections

    ## Submits a bid.
    # @param bid Bid to submit.
    # @param day Key of the day in which the bid is valid.
    def submit(self, bid, day=0):
        if day not in self.orders:
            self.orders[day] = OrdersBook()
        self.orders[day].append(bid)

    ## Clears the market.
    # @param day Key of the day.
    def clearing(self, day=0):
        if DEBUG:
            print("Clear DAM.")

        # Obtain the orders book
        book = self.orders[day]

        # Create the optimization model
        model = ConcreteModel()
        model.periods = Set(initialize=book.periods)
        model.bids = Set(initialize=range(len(book.bids)))
        model.L = Set(initialize=book.locations)
        model.sBids = Set(initialize=[i for i in range(len(book.bids)) if type(book.bids[i]) is SinglePeriodBid])
        model.bBids = Set(initialize=[i for i in range(len(book.bids)) if type(book.bids[i]) is BlockBid])
        model.C = Set(initialize=[c for c in self.connections.keys()]+[(c[1], c[0]) for c in self.connections.keys()])

        # Variables
        model.xs = Var(model.sBids, domain=Reals, bounds=(0.0, 1.0))  # Single period bids acceptance
        model.xb = Var(model.bBids, domain=Binary)  # Block bids acceptance
        model.pi = Var(model.L * model.periods, domain=Reals, bounds=self.priceCap)  # Market prices
        model.s = Var(model.bids, domain=NonNegativeReals)  # Bids
        def flowBounds(m, l1, l2, t):
            capacity = 0
            try:
                capacity = self.connections[(l1, l2)]
            except KeyError:
                pass
            try:
                capacity = self.connections[(l2, l1)]
            except KeyError:
                pass
            return (0, capacity)
        model.f = Var(model.C * model.periods, domain=NonNegativeReals, bounds=flowBounds)
        model.u = Var(model.C * model.periods, domain=NonNegativeReals)

        # Objective
        def primalObj(m):
            # Single period bids cost
            expr = summation({i: book.bids[i].cost*book.bids[i].volume for i in m.sBids}, m.xs)
            # Block bids cost
            expr += summation({i: book.bids[i].cost*sum(book.bids[i].volumes.values()) for i in m.bBids}, m.xb)
            return -expr
        model.obj = Objective(rule=primalObj, sense=maximize)

        # Balance constraint
        balanceExpr = {l: {t: 0.0 for t in model.periods} for l in model.L}
        for i in model.sBids:
            bid = book.bids[i]
            balanceExpr[bid.location][bid.period] += bid.volume*model.xs[i]
        for i in model.bBids:
            bid = book.bids[i]
            for t, v in bid.volumes.items():
                balanceExpr[bid.location][t] += v*model.xb[i]

        def balanceCstr(m, l, t):
            export = 0.0
            for c in model.C:
                if c[0] == l:
                    export += model.f[c, t]
                elif c[1] == l:
                    export -= model.f[c, t]
            return balanceExpr[l][t] == export
        model.balance = Constraint(model.L * book.periods, rule=balanceCstr)

        # Surplus of single period bids
        def sBidSurplus(m, i):
            bid = book.bids[i]
            return m.s[i] + m.pi[bid.location, bid.period]*-bid.volume >= bid.cost*-bid.volume
        model.sBidSurplus = Constraint(model.sBids, rule=sBidSurplus)

        # Surplus of block bids
        def bBidSurplus(m, i):
            bid = book.bids[i]
            bidVolume = -sum(bid.volumes.values())
            bigM = (self.priceCap[1]-self.priceCap[0])*bidVolume
            return m.s[i] + sum([m.pi[bid.location, t]*-v for t, v in bid.volumes.items()]) >= bid.cost*bidVolume + bigM*(1-m.xb[i])
        model.bBidSurplus = Constraint(model.bBids, rule=bBidSurplus)

        # Dual connections capacity
        def dualCapacity(m, l1, l2, t):
            exportPrices = 0.0
            for l in m.L:
                if l == l1:
                    exportPrices += m.pi[l, t]
                elif l == l2:
                    exportPrices -= m.pi[l, t]
            return m.u[l1, l2, t] + exportPrices >= 0.0
        model.dualCapacity = Constraint(model.C * model.periods, rule=dualCapacity)

        # Dual optimality
        def dualOptimality(m):
            dualObj = summation(m.s)
            for c in model.C:
                capacity = 0
                try:
                    capacity = self.connections[(c[0], c[1])]
                except KeyError:
                    pass
                try:
                    capacity = self.connections[(c[1], c[0])]
                except KeyError:
                    pass
                for t in m.periods:
                    dualObj += capacity * m.u[c, t]
            return primalObj(m) >= dualObj
        model.dualOptimality = Constraint(rule=dualOptimality)

        # Write .LP
        if DEBUG:
            model.write(filename="debug/damClearing.lp", format=ProblemFormat.cpxlp, io_options={"symbolic_solver_labels": True})
        # Solve
        SOLVER.solve(model, tee=True, keepfiles=True)
        if len(model.solutions) == 0:
            raise Exception('No solution found when clearing the day-ahead energy market.')

        # Load results
        if DEBUG:
            print("\twelfare: %s" % value(model.obj))
            print("Flow dual prices:")
        for t in model.periods:
            for c in model.C:
                print("%s -> %s : %.2f" % (c[0], c[1], model.u[c, t].value))
        self._buildSolution(model, book)

    ## Store the solution of the day-ahead market in the order book.
    # @param model Solved model.
    # @param book Orders book.
    def _buildSolution(self, model, book):
        book.volumes = {l: {t: 0.0 for t in book.periods} for l in model.L}
        book.prices = {l: {t: round(model.pi[l, t].value, 2) for t in book.periods} for l in model.L}
        for i in model.sBids:
            bid = book.bids[i]

            # Obtain and save the volume
            xs = model.xs[i].value
            bid.acceptance = xs

            # Update volumes and prices
            if xs > EPS and bid.volume > 0:
                t = bid.period

                # Compute the total volumes exchanged
                book.volumes[bid.location][t] += bid.volume*xs

        for i in model.bBids:
            bid = book.bids[i]

            # Obtain and save the volume
            xb = model.xb[i].value
            bid.acceptance = model.xb[i].value

            if xb > EPS:
                for t, v in bid.volumes.items():
                    book.volumes[bid.location][t] += v

    ## Get the cleared volumes.
    # @param day Key of the day.
    # @param location Location.
    def volumes(self, day=0, location=None):
        return self.orders[day].volumes[location]


    ## Get the system marginal prices.
    # @param day Key of the day.
    # @param location Location.
    def prices(self, day=0, location=None):
        return self.orders[day].prices[location]

    ## @var orders
    # Dictionary taking as key the day and as values the order books.
    ## @var priceCap
    # Tuple with the minimum and maximum price.
    ## @var connections
    # List of connections between the locations.
    # Dictionary where the keys are tuple of locations and values the capacity.

## Bid.
class Bid:
    ## Constructor.
    # @param owner Owner of the bid.
    # @param location Location.
    def __init__(self, owner=None, location=None):
        self.owner = owner
        self.acceptance = None
        self.location = location

    ## @var owner
    # Global identifier of the owner of the bid.
    ## @var acceptance
    # Acceptance of the bid in [0,1].
    ## @var location
    # Location.

## Single period bid of the energy market.
class SinglePeriodBid(Bid):
    ## Constructor.
    # @param volume Volume of the bid.
    # @param cost Cost per unit of volume.
    # @param period Period of the bid.
    # @param owner Owner of the bid.
    # @param location Location.
    def __init__(self, volume=0.0, cost=0.0, period=0, owner=None, location=None):
        Bid.__init__(self, owner=owner, location=location)
        self.volume = volume
        self.cost = cost
        self.period = period

    ## @var volume
    # Volume of the bid.
    ## @var cost
    # Marginal cost of the bid.
    ## @var period
    # Period of concern of the bid.


## Block bid of the energy market.
class BlockBid(Bid):
    ## Constructor.
    # @param volumes Volumes dictionary where the key are periods and values the volumes.
    # @param cost Cost of the bid.
    # @param owner Owner of the bid.
    # @param location Location.
    def __init__(self, volumes={}, cost=0.0, owner=None, location=None):
        Bid.__init__(self, owner=owner)
        self.volumes = volumes
        self.cost = cost
        self.location = location

    ## @var volumes
    # Volumes dictionary where the key are periods and values the volumes.
    ## @var cost
    # Cost of the bid.
    # Period of concern of the bid.
    ## @var location
    # Location.


## Orders book of bids.
class OrdersBook:
    ## Constructor.
    def __init__(self):
        self.bids = []
        self.periods = Set()
        self.prices = None
        self.locations = Set()

    ## Append a bid to the orders book.
    # @param bid Bid.
    def append(self, bid):
        self.bids.append(bid)

        # Update the sets
        if bid.location not in self.locations:
            self.locations.add(bid.location)

        if type(bid) is SinglePeriodBid:
            if bid.period not in self.periods:
                self.periods.add(bid.period)
        elif type(bid) is BlockBid:
            for t in bid.volumes.keys():
                if t not in self.periods:
                    self.periods.add(t)

    ## @var bids
    # List of bids.
    ## @var periods
    # Number of periods.
    ## @var prices
    # System Marginal prices.

# Starting point from python #
if __name__ == "__main__":

    #PREPROCESS
    #----------

    index_day = data_f.day

    index_ROW = 3 # MAX index 8

    QuantityROW = [6000,6500, 7000, 7500, 8000, 8500, 9000,9500,10000]
    print("")
    print("Day %s" %index_day)
    print("Quantity_ROW : %s" %QuantityROW[index_ROW]) #ROW = Rest of the world
    print("")

    #Offers - Renewable

    renewable_prod_day_ahead = np.zeros(data_f.nperiods)
    renewable_prod_measured = np.zeros(data_f.nperiods)
    renewable_imbalance = np.zeros(data_f.nperiods)
    for i in range(data_f.nperiods):
        renewable_prod_day_ahead[i] = data_w.day_ahead_wind[index_day][i] + data_s.day_ahead_sun[index_day][i]
        renewable_prod_measured[i] = data_w.measured_wind[index_day][i] + data_s.measured_sun[index_day][i]
        renewable_imbalance[i] = (data_w.measured_wind[index_day][i] + data_s.measured_sun[index_day][i]) - (data_w.day_ahead_wind[index_day][i] + data_s.day_ahead_sun[index_day][i])
        #This variable is positive if the data measured is higher than the quantity bid, we must use down regulation
        #              is negative if the data measured is lower than the quantity bid, we must use up regulation



    #Decomposition of the imbalance quantity

    renewable_imbalance_pos = np.zeros(data_f.nperiods)
    renewable_imbalance_neg = np.zeros(data_f.nperiods)

    for i in range(data_f.nperiods):
        if renewable_imbalance[i] > 0: 
            renewable_imbalance_pos[i] = abs(renewable_imbalance[i])
            renewable_imbalance_neg[i] = 0
        else:
            renewable_imbalance_pos[i] = 0
            renewable_imbalance_neg[i] = abs(renewable_imbalance[i])





    #Offers - Rest of the world

    QuantityBidROW1 = np.zeros(data_f.nperiods) 
    PriceROW1 = np.zeros(data_f.nperiods) 
    QuantityBidROW2 = np.zeros(data_f.nperiods) 
    PriceROW2 = np.zeros(data_f.nperiods) 

    for i in range(data_f.nperiods):
        #First part
        QuantityBidROW1[i] = QuantityROW[index_ROW]
        PriceROW1[i] = 20
        #Second part
        QuantityBidROW2[i] = 10000
        PriceROW2[i] = 70



    #Demand
    QuantityDemandTot = np.zeros(data_f.nperiods) 
    PriceDemand = np.zeros(data_f.nperiods) 

    for i in range(data_f.nperiods):
        QuantityDemandTot[i] = -data_d.volume_demand[20][i]
        PriceDemand[i] = 100 


    #-------------------
    # CLEARING DAY-AHEAD
    #-------------------
    dam = DAM()

    # Demand bids
    #------------
    for i in range(data_f.nperiods):
        dam.submit(SinglePeriodBid(QuantityDemandTot[i], PriceDemand[i], i))

    # Offer bids 
    #-----------
    # Flexible power plant
    Bids = []

    for i in range(data_f.nbids):
            if data_f.TypeBid[i] == 1:
                Bids.append(SinglePeriodBid(data_f.QuantityBid[i], data_f.PriceBid[i], data_f.TimestepBid[i]))
            elif data_f.TypeBid[i] == 2:
                Bids.append(BlockBid({(data_f.TimestepBid[i]):(data_f.QuantityBid[i])}, data_f.PriceBid[i]))

    
    for i in range(data_f.nbids):
        dam.submit(Bids[i])


    #Renewable & Rest of the world
    R_Bids = []
    ROW1_Bids = []
    ROW2_Bids = []

    for i in range(data_f.nperiods):
        R_Bids.append(SinglePeriodBid(renewable_prod_day_ahead[i], 0.0, i))
        ROW1_Bids.append(SinglePeriodBid(QuantityBidROW1[i], PriceROW1[i], i))
        ROW2_Bids.append(SinglePeriodBid(QuantityBidROW2[i], PriceROW2[i], i))
    for i in range(data_f.nperiods):
        dam.submit(R_Bids[i])
        dam.submit(ROW1_Bids[i])
        dam.submit(ROW2_Bids[i])





    # Clear the market
    #-----------------
    dam.clearing()


    #POST-PROCESSING
    #---------------
    #---------------

    print("")
    print("Prices: %s" % dam.prices())
    print("Volume: %s" % dam.volumes())
    print("")

    #------------------
    # LOSS OF EARNINGS
    #------------------

    Loss_time_step  = np.zeros(data_f.nbids)

    Loss = 0

    for i in range(data_f.nbids):
        Loss_time_step[i] = (1-Bids[i].acceptance)*data_f.QuantityBid[i]*data_p.price[index_day][int(math.ceil(data_f.TimestepBid[i]/4)-1)]*0.25

    Loss = np.sum(Loss_time_step)

    print("Total loss: %s" % Loss)


    #------------------
    # RENEWABLE PROFIT
    #------------------


    Renewable_profit_time_step  = np.zeros(data_f.nperiods)
    
    for i in range(data_f.nperiods):
        Renewable_profit_time_step[i] = 0.25*renewable_prod_day_ahead[i]*data_p.price[index_day][int(math.floor(i/4))] + 0.25*renewable_imbalance_pos[i] * data_i.up_price[index_day][i] - 0.25*renewable_imbalance_neg[i] * data_i.down_price[index_day][i]

        
    Renewable_profit = np.sum(Renewable_profit_time_step)

    Compensation_renewable_profit_time_step  = np.zeros(data_f.nperiods)
    Perfect_renewable_profit_time_step  = np.zeros(data_f.nperiods)

    for i in range(data_f.nperiods):
        Perfect_renewable_profit_time_step[i] = 0.25*renewable_prod_measured[i]*data_p.price[index_day][int(math.floor(i/4))]
        Compensation_renewable_profit_time_step[i] = 0.25*renewable_prod_measured[i]*data_p.price[index_day][int(math.floor(i/4))]- Renewable_profit_time_step[i]

    Perfect_renewable_profit = np.sum(Perfect_renewable_profit_time_step)
    Compensation_renewable_profit = np.sum(Compensation_renewable_profit_time_step)

    print("Renewable profit: %s" %Renewable_profit)
    print("Perfect profit: %s" %Perfect_renewable_profit)
    print("Loss_wrong_pred: %s" %Compensation_renewable_profit)



    #Writing losses in an EXCEL file
    workbook = openpyxl.load_workbook('data/Loss.xlsx')
    sheet = workbook.get_sheet_by_name('day_ahead')
    lol = sheet.cell(row = index_day+1, column = index_ROW+1) 
    lol.value = Loss
    workbook.save('data/Loss.xlsx')

    
