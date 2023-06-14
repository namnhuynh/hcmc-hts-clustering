import pandas as pd
import pickle
import visualisation as vis

# htsFile = ['../../../data/HIS/HIS_1st100_translated.xlsx','Sheet2']
htsFile = {'filename': '../../../data/HIS/data HIS_translated.xlsx',
           'sheet_name': 'Form_02（ALLmember)_Vie'}
htsColumns = {1: 'hhID', 2: 'indName', 3: 'indivID', 4: 'age', 5: 'gender', 6: 'eduLevel',
              7: 'licence', 8: 'ownCar', 9: 'ownMotor', 10: 'ownBike', 11: 'ownEbike', 12: 'travelShareType',
              13: 'isImpaired', 14: 'isHholdHead', 15: 'residencyStatus', 16: 'isHousemaid',
              17: 'jobType', 18: 'industry', 19: 'businessType', 20: 'employStatus', 21: 'monthlyIncome',
              # 22: 'tzWork',
              # 23: 'hhWorkFr', 24: 'mmWorkFr', 25: 'ddWorkFr', 26: 'hhWorkTo', 27: 'mmWorkto', 28: 'ddWorkTo',
              # 29: 'workTravMode',
              # 30: 'toWorkTravTime', 31: 'toWorkTravSatis', 32: 'toWorkTravTimeMax',
              # 33: 'toHomeTravTime', 34: 'toHomeTravSatis', 35: 'toHomeTravTimeMax',
              # 36: 'tripsType',  # trips from yesterday (1) or trips from a representative day (2)

              37: 'tzOrigTrip1', 38: 'typeOrigTrip1', 39: 'hhDepTimeTrip1', 40: 'mmDepTimeTrip1',
              41: 'tzDestTrip1', 42: 'typeDestTrip1',  # 43: 'hhArrTimeTrip1', 44: 'mmArrTimeTrip1',
              45: 'purposeTrip1',
              # 46: 'modeTp1Trip1', 47: 'nMinsTp1Trip1', 48: 'fareTp1Trip1', 49: 'tzTp1Trip1',
              # 50: 'modeTp2Trip1', 51: 'nMinsTp2Trip1', 52: 'fareTp2Trip1', 53: 'tzTp2Trip1',
              # 54: 'modeTp3Trip1', 55: 'nMinsTp3Trip1', 56: 'fareTp3Trip1', 57: 'tzTp3Trip1',
              58: 'modeTrip1', 59: 'nMinsTrip1',
              # 60: 'fareTrip1', 61: 'parkTypeTrip1', 62: 'parkCostTrip1', 63: 'modeChoiceReasonTrip1', 64: 'satisTrip1',

              65: 'tzOrigTrip2', 66: 'typeOrigTrip2', 67: 'hhDepTimeTrip2', 68: 'mmDepTimeTrip2',
              69: 'tzDestTrip2', 70: 'typeDestTrip2',  # 71: 'hhArrTimeTrip2', 72: 'mmArrTimeTrip2',
              73: 'purposeTrip2',
              # 74: 'modeTp1Trip2', 75: 'nMinsTp1Trip2', 76: 'fareTp1Trip2', 77: 'tzTp1Trip2',
              # 78: 'modeTp2Trip2', 79: 'nMinsTp2Trip2', 80: 'fareTp2Trip2', 81: 'tzTp2Trip2',
              # 82: 'modeTp3Trip2', 83: 'nMinsTp3Trip2', 84: 'fareTp3Trip2', 85: 'tzTp3Trip2',
              86: 'modeTrip2', 87: 'nMinsTrip2',
              # 88: 'fareTrip2', 89: 'parkTypeTrip2', 90: 'parkCostTrip2', 91: 'modeChoiceReasonTrip2', 92: 'satisTrip2',

              93: 'tzOrigTrip3', 94: 'typeOrigTrip3', 95: 'hhDepTimeTrip3', 96: 'mmDepTimeTrip3',
              97: 'tzDestTrip3', 98: 'typeDestTrip3',  # 99: 'hhArrTimeTrip3', 100: 'mmArrTimeTrip3',
              101: 'purposeTrip3',
              # 102: 'modeTp1Trip3', 103: 'nMinsTp1Trip3', 104: 'fareTp1Trip3', 105: 'tzTp1Trip3',
              # 106: 'modeTp2Trip3', 107: 'nMinsTp2Trip3', 108: 'fareTp2Trip3', 109: 'tzTp2Trip3',
              # 110: 'modeTp3Trip3', 111: 'nMinsTp3Trip3', 112: 'fareTp3Trip3', 113: 'tzTp3Trip3',
              114: 'modeTrip3', 115: 'nMinsTrip3',
              # 116: 'fareTrip3', 117: 'parkTypeTrip3', 118: 'parkCostTrip3', 119: 'modeChoiceReasonTrip3',
              # 120: 'satisTrip3',

              121: 'tzOrigTrip4', 122: 'typeOrigTrip4', 123: 'hhDepTimeTrip4', 124: 'mmDepTimeTrip4',
              125: 'tzDestTrip4', 126: 'typeDestTrip4',  # 127: 'hhArrTimeTrip4', 128: 'mmArrTimeTrip4',
              129: 'purposeTrip4',
              # 130: 'modeTp1Trip4', 131: 'nMinsTp1Trip4', 132: 'fareTp1Trip4', 133: 'tzTp1Trip4',
              # 134: 'modeTp2Trip4', 135: 'nMinsTp2Trip4', 136: 'fareTp2Trip4', 137: 'tzTp2Trip4',
              # 138: 'modeTp3Trip4', 139: 'nMinsTp3Trip4', 140: 'fareTp3Trip4', 141: 'tzTp3Trip4',
              142: 'modeTrip4', 143: 'nMinsTrip4',
              # 144: 'fareTrip4', 145: 'parkTypeTrip4', 146: 'parkCostTrip4', 147: 'modeChoiceReasonTrip4',
              # 148: 'satisTrip4',

              149: 'tzOrigTrip5', 150: 'typeOrigTrip5', 151: 'hhDepTimeTrip5', 152: 'mmDepTimeTrip5',
              153: 'tzDestTrip5', 154: 'typeDestTrip5',  # 155: 'hhArrTimeTrip5', 156: 'mmArrTimeTrip5',
              157: 'purposeTrip5',
              # 158: 'modeTp1Trip5', 159: 'nMinsTp1Trip5', 160: 'fareTp1Trip5', 161: 'tzTp1Trip5',
              # 162: 'modeTp2Trip5', 163: 'nMinsTp2Trip5', 164: 'fareTp2Trip5', 165: 'tzTp2Trip5',
              # 166: 'modeTp3Trip5', 167: 'nMinsTp3Trip5', 168: 'fareTp3Trip5', 169: 'tzTp3Trip5',
              170: 'modeTrip5', 171: 'nMinsTrip5',
              # 172: 'fareTrip5', 173: 'parkTypeTrip5', 174: 'parkCostTrip5', 175: 'modeChoiceReasonTrip5',
              # 176: 'satisTrip5',

              177: 'tzOrigTrip6', 178: 'typeOrigTrip6', 179: 'hhDepTimeTrip6', 180: 'mmDepTimeTrip6',
              181: 'tzDestTrip6', 182: 'typeDestTrip6',  # 183: 'hhArrTimeTrip6', 184: 'mmArrTimeTrip6',
              185: 'purposeTrip6',
              # 186: 'modeTp1Trip6', 187: 'nMinsTp1Trip6', 188: 'fareTp1Trip6', 189: 'tzTp1Trip6',
              # 190: 'modeTp2Trip6', 191: 'nMinsTp2Trip6', 192: 'fareTp2Trip6', 193: 'tzTp2Trip6',
              # 194: 'modeTp3Trip6', 195: 'nMinsTp3Trip6', 196: 'fareTp3Trip6', 197: 'tzTp3Trip6',
              198: 'modeTrip6', 199: 'nMinsTrip6',
              # 200: 'fareTrip6', 201: 'parkTypeTrip6', 202: 'parkCostTrip6', 203: 'modeChoiceReasonTrip6',
              # 204: 'satisTrip6',

              205: 'tzOrigTrip7', 206: 'typeOrigTrip7', 207: 'hhDepTimeTrip7', 208: 'mmDepTimeTrip7',
              209: 'tzDestTrip7', 210: 'typeDestTrip7',  # 211: 'hhArrTimeTrip7', 212: 'mmArrTimeTrip7',
              213: 'purposeTrip7',
              # 214: 'modeTp1Trip7', 215: 'nMinsTp1Trip7', 216: 'fareTp1Trip7', 217: 'tzTp1Trip7',
              # 218: 'modeTp2Trip7', 219: 'nMinsTp2Trip7', 220: 'fareTp2Trip7', 221: 'tzTp2Trip7',
              # 222: 'modeTp3Trip7', 223: 'nMinsTp3Trip7', 224: 'fareTp3Trip7', 225: 'tzTp3Trip7',
              226: 'modeTrip7', 227: 'nMinsTrip7',
              # 228: 'fareTrip7', 229: 'parkTypeTrip7', 230: 'parkCostTrip7', 231: 'modeChoiceReasonTrip7',
              # 232: 'satisTrip7',
              }

genderMeanings = {1: 'male', 2: 'female'}

eduLevelMeanings = {1: 'elementary', 2: 'juniorHigh', 3: 'seniorHigh', 4: 'vocationalTraining', 5: 'diploma/Bachelor',
                    6: 'postgrad', 7: 'NA'}
eduLevelGrouping = {'elementary': 'juniorHighOrLower', 'juniorHigh': 'juniorHighOrLower',
                    'seniorHigh': 'seniorHighAndEquivalent', 'vocationalTraining': 'seniorHighAndEquivalent',
                    'diploma/Bachelor': 'tertiaryEduOrHigher', 'postgrad': 'tertiaryEduOrHigher',
                    'NA': 'NA'}

licenceTypeMeanings = {1: 'Motorbike', 2: 'Automobile', 3: 'Both', 4: 'None'}
licenceTypeGrouping = {'Motorbike': 'yes', 'Automobile': 'yes', 'Both': 'yes', 'None': 'no'}

ownCarMeanings = {1: 'Yes', 2: 'No'}  # combine with ownMotor
ownMotorMeanings = {1: 'Yes', 2: 'No'}  # combine with ownCar
ownBikeMeanings = {1: 'Yes', 2: 'No'}  # drop
ownEbikeMeanings = {1: 'Yes', 2: 'No'}  # drop

travelShareTypeMeanings = {1: 'Ride-share with family members', 2: 'Ride-share with non-family members',
                           3: 'public transport', 4: 'Others'}  # drop

isImpairedMeanings = {1: 'Yes', 2: 'No'}  # drop
isHholdHeadMeanings = {1: 'Yes', 2: 'No'}  # drop
residencyStatusMeanings = {1: 'Permanent resident', 2: 'Long term temporary resident',
                           3: 'Short term temporary resident', 4: 'No registration'}  # drop
isHousemaidMeanings = {1: 'Yes', 2: 'No'}  # drop

jobTypeMeanings = {1: 'Manager', 2: 'Office worker', 3: 'Skilled worker', 4: 'Labour worker', 5: 'Handicraft worker',
                   6: 'Grocery worker', 7: 'Army/Police', 8: '(Personal) driver', 9: 'Housemaid', 10: 'Other',
                   11: 'Students (working parttime)', 12: 'Students (not working)', 13: 'Housewife',
                   14: 'Unemployed/Retired'}
jobTypeGrouping = {'Manager': 'officeWorker',
                   'Office worker': 'officeWorker',
                   'Skilled worker': 'labourWorker',
                   'Labour worker': 'labourWorker',
                   'Handicraft worker': 'labourWorker',
                   'Grocery worker': 'labourWorker',
                   '(Personal) driver': 'labourWorker',
                   'Housemaid': 'labourWorker',
                   'Other': 'other',
                   'Army/Police': 'other',
                   'Students (working parttime)': 'student',
                   'Students (not working)': 'student',
                   'Housewife': 'unemployed/Retired',
                   'Unemployed/Retired': 'unemployed/Retired'}

industryMeanings = {1: 'Agri- or aqua-culture', 2: 'manufacturing', 3: 'construction', 4: 'trading and hospitality',
                    5: 'Public sector or army', 6: 'Transportation, Logistics, Communication',
                    7: 'Finance, insurance, real estate', 8: 'Education', 9: 'Social services (e.g. healthcare)',
                    10: 'Other'}  # drop
businessTypeMeanings = {1: 'State-owned (central government)', 2: 'State-owned (provincial government)',
                        3: 'Private (domestic)', 4: 'Private (partly foreign-owned)',
                        5: 'Private (foreign-owned)'}  # drop

employStatusMeanings = {1: 'Permanent fulltime', 2: 'Short-term contracts', 3: 'Business owner', 4: 'Freelancer'}
employStatusGrouping = {'Permanent fulltime': 'permanentFulltime',
                        'Business owner': 'businessOwner',
                        'Short-term contracts': 'short-termOrFreelancer',
                        'Freelancer': 'short-termOrFreelancer'}

monthlyIncomeMeanings = {1: '1. under200,000', 2: '2. 201,000 – 400,000', 3: '3. 401,000 – 800,000',
                         4: '4. 801,000 – 1,500,000', 5: '5. 1,501,000 – 2,500,000', 6: '6. 2,501,000 – 4,000,000',
                         7: '7. 4,001,000 – 6,000,000', 8: '8. 6,001,000 – 8,000,000', 9: '9. 8,001,000 – 10,000,000',
                         10: '10. 10,001,000 – 15,000,000', 11: '11. 15,001,000 – 20,000,000',
                         12: '12. 20,001,000 – 25,000,000', 13: '13. 25,001,000 – 30,000,000',
                         14: '14. 30,001,000 – 50,000,000', 15: '15. Over50,000,000'}
monthlyIncomeGrouping = {'1. under200,000': 'under2.5mVND',  # '1.under2,500,000',
                         '2. 201,000 – 400,000': 'under2.5mVND',  # '1.under2,500,000',
                         '3. 401,000 – 800,000': 'under2.5mVND',  # '1.under2,500,000',
                         '4. 801,000 – 1,500,000': 'under2.5mVND',  # '1.under2,500,000',
                         '5. 1,501,000 – 2,500,000': 'under2.5mVND',  # '1.under2,500,000',
                         '6. 2,501,000 – 4,000,000': '2.5mVND-4.0mVND',  # '2.2,501,000 – 4,000,000',
                         '7. 4,001,000 – 6,000,000': '4.0mVND-6.0mVND',  # '3.4,001,000 – 6,000,000',
                         '8. 6,001,000 – 8,000,000': '6.0mVND-8.0mVND',  # '4.6,001,000 – 8,000,000',
                         '9. 8,001,000 – 10,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '10. 10,001,000 – 15,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '11. 15,001,000 – 20,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '12. 20,001,000 – 25,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '13. 25,001,000 – 30,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '14. 30,001,000 – 50,000,000': 'over8.0mVND',  # '5.over8,000,000',
                         '15. Over50,000,000': 'over8.0mVND'}  # '5.over8,000,000'}

tripPurposeMeanings = {1: 'toHome', 2: 'toWork', 3: 'toSchool', 4: 'personalBusiness', 5: 'businessTrips',
                       6: 'eatingOut', 7: 'socialActivities', 8: 'shopping', 9: 'pickingSomeone',
                       10: 'seeingAround', 11: 'others'}
tripPurposeGrouping = {'toHome': 'toHome',
                       'toWork': 'toWork',
                       'businessTrips': 'toWork',
                       'toSchool': 'toSchool',
                       'personalBusiness': 'socialActivities',
                       'eatingOut': 'socialActivities',
                       'socialActivities': 'socialActivities',
                       'shopping': 'socialActivities',
                       'pickingSomeone': 'socialActivities',
                       'seeingAround': 'socialActivities',
                       'others': 'socialActivities'}

tripModeMeanings = {1: 'Walk (100-500 metres)', 2: 'Walk (over 500 metres)', 3: 'Bicycle rider', 4: 'Bicycle passenger',
                    5: 'Electric bicycle', 6: 'motobike rider', 7: 'motobike passenger', 8: 'car driver',
                    9: 'car passenger', 10: 'taxi', 11: 'coach services', 12: 'school bus', 13: 'company bus',
                    14: 'small bus (under 25 passengers)', 15: 'bus (over 25 passengers)', 16: 'motobike taxi',
                    17: 'tricycle', 18: 'motorised tricycle', 19: 'Ute', 20: 'Truck', 21: 'River ferry',
                    22: 'Other ferries', 23: 'Rail', 24: 'Air', 25: 'Other'}

tripModeGrouping = {'Walk (100-500 metres)': 'walk',
                    'Walk (over 500 metres)': 'walk',
                    'Bicycle rider': 'bicycle',
                    'Bicycle passenger': 'bicycle',
                    'Electric bicycle': 'bicycle',
                    'motobike rider': 'motorbike',
                    'motobike passenger': 'motorbike',
                    'car driver': 'car',
                    'car passenger': 'car',
                    'taxi': 'car',
                    'coach services': 'bus',
                    'school bus': 'bus',
                    'company bus': 'bus',
                    'small bus (under 25 passengers)': 'bus',
                    'bus (over 25 passengers)': 'bus',
                    'motobike taxi': 'motorbike',
                    'tricycle': 'bicycle',
                    'motorised tricycle': 'motorbike',
                    'Ute': 'other',
                    'Truck': 'other',
                    'River ferry': 'other',
                    'Other ferries': 'other',
                    'Rail': 'other',
                    'Air': 'other',
                    'Other': 'other'}

odTypeMeanings = {1: 'Home', 2: 'Work', 3: 'Factory/warehouse', 4: 'School', 5: 'supermarket/mall',
                  6: 'recreational venues', 7: 'healthcare', 8: 'restaurants', 9: 'others'}

odTypeGrouping = {'Home': 'home',
                  'Work': 'work',
                  'Factory/warehouse': 'work',
                  'School': 'school',
                  'supermarket/mall': 'amenities',
                  'recreational venues': 'amenities',
                  'healthcare': 'amenities',
                  'restaurants': 'amenities',
                  'others': 'others'}

district2tz = {'District1': [i for i in range(1, 12 + 1)],
               'District3': [i for i in range(13, 24 + 1)],
               'District4': [i for i in range(25, 32 + 1)],
               'District5': [i for i in range(33, 47 + 1)],
               'District6': [i for i in range(48, 59 + 1)],
               'District8': [i for i in range(60, 72 + 1)],
               'District10': [i for i in range(73, 84 + 1)],
               'District11': [i for i in range(85, 97 + 1)],
               'GoVap': [i for i in range(98, 105 + 1)],
               'BìnhThanh': [i for i in range(106, 113 + 1)],
               'TanBinh': [i for i in range(114, 122 + 1)],
               'TanPhu': [i for i in range(123, 129 + 1)],
               'PhuNhuan': [i for i in range(130, 136 + 1)],
               'District2': [i for i in range(137, 141 + 1)],
               'District7': [i for i in range(142, 147 + 1)],
               'District9': [i for i in range(148, 156 + 1)],
               'District12': [i for i in range(157, 163 + 1)],
               'NhaBe': [i for i in range(164, 170 + 1)],
               'BinhChanh': [i for i in range(171, 179 + 1)] + [i for i in range(184, 190 + 1)],
               'BinhTan': [i for i in range(180, 183 + 1)],
               'HocMon': [i for i in range(191, 200 + 1)],
               'ThuDuc': [i for i in range(201, 209 + 1)],
               'CuChi': [i for i in range(210, 215 + 1)],
               'CanGio': [216],
               'LongAn_CanGiuoc': [219, 220, 221],
               'LongAn_CanDuoc': [223],
               'LongAn_BenLuc': [226, 227, 228],
               'LongAn_DucHoa': [231, 232, 233],
               'BinhDuong_ThuDauMot': [235, 236, 237, 238],
               'BinhDuong_ThuanAn': [239, 240],
               'BinhDuong_DiAn': [241, 242],
               'DongNai_BienHoa': [i for i in range(248, 258 + 1)],
               'DongNai_NhonTrach': [263, 264, 265]}

missingTZs = [0,
              217, 218,
              222,
              224, 225,
              229, 230,
              234,
              243, 244, 245, 246, 247,
              259, 260, 261, 262]


# ======================================================================================================================
def readHTS(readFromPkl=True):
    if readFromPkl:
        with open('../../../data/HIS/data HIS_translated.pkl', 'rb') as f:
            dfHTS = pickle.load(f)
    else:
        nrows = 60551
        dfHTS = pd.read_excel(io=htsFile['filename'], sheet_name=htsFile['sheet_name'],
                              skiprows=[1, 2, 3, 4], nrows=nrows)

        # drops columns with 'Unnamed' in them
        unnamedCols = [x for x in dfHTS.columns if type(x) == str]
        dfHTS.drop(columns=unnamedCols, inplace=True)

        delCols1_0 = [43, 44]  # hh and mm of arrival time of trip 1
        delCols1_1 = [i for i in range(46, 57 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 1
        delCols1_2 = [i for i in range(60, 64 + 1)]  # columns related to fares, parking, satisfaction of trip 1

        delCols2_0 = [71, 72]  # hh and mm of arrival time of trip 2
        delCols2_1 = [i for i in range(74, 85 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 2
        delCols2_2 = [i for i in range(88, 92 + 1)]  # columns related to fares, parking, satisfaction of trip 2

        delCols3_0 = [99, 100]  # hh and mm of arrival time of trip 3
        delCols3_1 = [i for i in range(102, 113 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 3
        delCols3_2 = [i for i in range(116, 120 + 1)]  # columns related to fares, parking, satisfaction of trip 3

        delCols4_0 = [127, 128]  # hh and mm of arrival time of trip 4
        delCols4_1 = [i for i in range(130, 141 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 4
        delCols4_2 = [i for i in range(144, 148 + 1)]  # columns related to fares, parking, satisfaction of trip 4

        delCols5_0 = [155, 156]  # hh and mm of arrival time of trip 5
        delCols5_1 = [i for i in range(158, 169 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 5
        delCols5_2 = [i for i in range(172, 176 + 1)]  # columns related to fares, parking, satisfaction of trip 5

        delCols6_0 = [183, 184]  # hh and mm of arrival time of trip 6
        delCols6_1 = [i for i in range(186, 197 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 6
        delCols6_2 = [i for i in range(200, 204 + 1)]  # columns related to fares, parking, satisfaction of trip 6

        delCols7_0 = [211, 212]  # hh and mm of arrival time of trip 7
        delCols7_1 = [i for i in range(214, 225 + 1)]  # columns related to 4 attributes of 3 transfer points of trip 7
        delCols7_2 = [i for i in range(228, 232 + 1)]  # columns related to fares, parking, satisfaction of trip 7

        delCols8 = [i for i in range(233, 339 + 1)]  # columns at the end of the dataset that have no use.

        delCols9 = [i for i in range(22, 36 + 1)]  # columns related to work trips, redundant so removed

        delCols = delCols1_0 + delCols1_1 + delCols1_2 + \
                  delCols2_0 + delCols2_1 + delCols2_2 + \
                  delCols3_0 + delCols3_1 + delCols3_2 + \
                  delCols4_0 + delCols4_1 + delCols4_2 + \
                  delCols5_0 + delCols5_1 + delCols5_2 + \
                  delCols6_0 + delCols6_1 + delCols6_2 + \
                  delCols7_0 + delCols7_1 + delCols7_2 + \
                  delCols8 + delCols9
        dfHTS.drop(delCols, axis=1, inplace=True)  # drops them!

        dfHTS.rename(columns=htsColumns, inplace=True)  # rename the remaining columns
        dfHTS.fillna(-1, inplace=True)  # replace any NA or empty cells with -1
        dfHTS.replace(to_replace=' ', value=-1, inplace=True)
        # dfHTS.drop(columns=['indName'], inplace=True)  # drops individual names

        dfHTS.to_pickle('../../../data/HIS/data HIS_translated.pkl')

    return dfHTS


# ======================================================================================================================
def processDemographicAttributes(dfHTS):
    # translates numeric values to text
    demoCols = {'gender': genderMeanings,
                'eduLevel': eduLevelMeanings,
                'licence': licenceTypeMeanings,
                'ownCar': ownCarMeanings,
                'ownMotor': ownMotorMeanings,
                'ownBike': ownBikeMeanings,
                'ownEbike': ownEbikeMeanings,
                'travelShareType': travelShareTypeMeanings,
                'isImpaired': isImpairedMeanings,
                'isHholdHead': isHholdHeadMeanings,
                'residencyStatus': residencyStatusMeanings,
                'isHousemaid': isHousemaidMeanings,
                'jobType': jobTypeMeanings,
                'industry': industryMeanings,
                'businessType': businessTypeMeanings,
                'employStatus': employStatusMeanings,
                'monthlyIncome': monthlyIncomeMeanings}
    for col, meanings in demoCols.items():
        dfHTS[col] = dfHTS[col].map(meanings)
    dfHTS.to_csv('./tmpOutputs/dfHTS_original.csv', index=False)

    # plot original demographic attributes
    vis.plotDemographicCols(dfHTS, list(demoCols.keys()), 'demographics_original.html')

    # merges similar/related values
    dfHTS['eduLevel'] = dfHTS['eduLevel'].map(eduLevelGrouping)
    dfHTS['licence'] = dfHTS['licence'].map(licenceTypeGrouping)
    dfHTS['employStatus'] = dfHTS['employStatus'].map(employStatusGrouping)
    dfHTS['monthlyIncome'] = dfHTS['monthlyIncome'].map(monthlyIncomeGrouping)
    dfHTS['jobType'] = dfHTS['jobType'].map(jobTypeGrouping)

    def ownVehicle(row):
        if (row['ownCar'] == 'No') and (row['ownMotor'] == 'No'):
            return 'no'
        else:
            return 'yes'

    dfHTS['ownMotorVehicles'] = dfHTS.apply(lambda row: ownVehicle(row), axis=1)

    # drops demographic attributes that are presumably irrelevant to travel demands and/or behaviours
    cols2Drop = ['ownCar', 'ownMotor',  # merges into ownMotorVehicles
                 'ownBike',  # owning a bike wouldn't affect the way people travel
                 'ownEbike',  # same with owning an ebike
                 'travelShareType',  # likely irrelevant, shouldn't affect travel
                 'isImpaired',  # small variance, impaired has a very small proportion so ignored
                 'isHholdHead',  # being a household head or not shouldn't change how people travel
                 'residencyStatus',  # small variance, residency status shouldn't change how people travel anyway
                 'isHousemaid',  # small variance
                 'industry',  # 'Other' has highest frequency => uninformative, shouldn't affect travel anyway
                 'businessType'  # small variance, should not affect travel anyway
                 ]
    dfHTS.drop(columns=cols2Drop, inplace=True)
    vis.plotDemographicCols(dfHTS, list(demoCols.keys()) + ['ownMotorVehicles'], 'demographics_final.html')
    dfHTS.to_csv('./tmpOutputs/dfHTS_demoProcessed.csv', index=False)


# ======================================================================================================================
def processTripAttributes(dfHTS):
    def calcNumOfMinsFromZeroHr(row, hhCol, mmCol):
        if (row[hhCol] < 0) or (row[mmCol] < 0):
            return None
        return row[hhCol] * 60 + row[mmCol]

    # maps travel zones to districts
    tz2District = {}
    for district, zones in district2tz.items():
        for tz in zones:
            tz2District[tz] = district

    # converts all trip related columns to integer
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        dfHTS['tzOrigTrip%d' % iTrip] = dfHTS['tzOrigTrip%d' % iTrip].round(0).astype('int')
        dfHTS['tzDestTrip%d' % iTrip] = dfHTS['tzDestTrip%d' % iTrip].round(0).astype('int')
        dfHTS['typeOrigTrip%d' % iTrip] = dfHTS['typeOrigTrip%d' % iTrip].round(0).astype('int')
        dfHTS['typeDestTrip%d' % iTrip] = dfHTS['typeDestTrip%d' % iTrip].round(0).astype('int')
        dfHTS['purposeTrip%d' % iTrip] = dfHTS['purposeTrip%d' % iTrip].round(0).astype('int')
        dfHTS['modeTrip%d' % iTrip] = dfHTS['modeTrip%d' % iTrip].round(0).astype('int')
        dfHTS['hhDepTimeTrip%d' % iTrip] = dfHTS['hhDepTimeTrip%d' % iTrip].round(0).astype('int')
        dfHTS['mmDepTimeTrip%d' % iTrip] = dfHTS['mmDepTimeTrip%d' % iTrip].round(0).astype('int')
        dfHTS['nMinsTrip%d' % iTrip] = dfHTS['nMinsTrip%d' % iTrip].round(0).astype('int')

    # replaces strange numerical values
    dfHTS['typeOrigTrip3'] = dfHTS['typeOrigTrip3'].replace([11], 1)
    dfHTS.to_csv('./tmpOutputs/dfHTS_tripProcessed0.csv', index=False)

    # maps numerical values to text, then groups similar values
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        # groups origin types
        # dfHTS['typeOrigTrip%d' % iTrip] = dfHTS['typeOrigTrip%d' % iTrip].round(0).astype('int')
        dfHTS['typeOrigTrip%d' % iTrip] = dfHTS['typeOrigTrip%d' % iTrip].map(odTypeMeanings)
        dfHTS['typeOrigTrip%d' % iTrip] = dfHTS['typeOrigTrip%d' % iTrip].map(odTypeGrouping)
        # groups destination types
        # dfHTS['typeDestTrip%d' % iTrip] = dfHTS['typeDestTrip%d' % iTrip].round(0)#.astype('int')
        dfHTS['typeDestTrip%d' % iTrip] = dfHTS['typeDestTrip%d' % iTrip].map(odTypeMeanings)
        dfHTS['typeDestTrip%d' % iTrip] = dfHTS['typeDestTrip%d' % iTrip].map(odTypeGrouping)

        # groups trip purposes
        # dfHTS['purposeTrip%d' % iTrip] = dfHTS['purposeTrip%d' % iTrip].round(0)#.astype('int')
        dfHTS['purposeTrip%d' % iTrip] = dfHTS['purposeTrip%d' % iTrip].map(tripPurposeMeanings)
        dfHTS['purposeTrip%d' % iTrip] = dfHTS['purposeTrip%d' % iTrip].map(tripPurposeGrouping)

        # groups trip modes
        # dfHTS['modeTrip%d' % iTrip] = dfHTS['modeTrip%d' % iTrip].round(0)#.astype('int')
        dfHTS['modeTrip%d' % iTrip] = dfHTS['modeTrip%d' % iTrip].map(tripModeMeanings)
        dfHTS['modeTrip%d' % iTrip] = dfHTS['modeTrip%d' % iTrip].map(tripModeGrouping)

        # converts departure time and arrival times to number of minutes from start of day
        col_hhDepTimeTrip = 'hhDepTimeTrip%d' % iTrip
        col_mmDepTimeTrip = 'mmDepTimeTrip%d' % iTrip
        dfHTS['nMinsDepTimeTrip%d' % iTrip] = \
            dfHTS.apply(lambda row: calcNumOfMinsFromZeroHr(row, col_hhDepTimeTrip, col_mmDepTimeTrip), axis=1)
        dfHTS.drop(columns=[col_hhDepTimeTrip, col_mmDepTimeTrip], inplace=True)

        # maps origin and destination travel zones to origin and destination districts
        # dfHTS['tzOrigTrip%d' % iTrip] = dfHTS['tzOrigTrip%d' % iTrip].round(0)#.astype('int')
        dfHTS['origTrip%d' % iTrip] = dfHTS['tzOrigTrip%d' % iTrip].map(tz2District)
        # dfHTS['tzDestTrip%d' % iTrip] = dfHTS['tzDestTrip%d' % iTrip].round(0)#.astype('int')
        dfHTS['destTrip%d' % iTrip] = dfHTS['tzDestTrip%d' % iTrip].map(tz2District)

        # replaces value -1 in nMinsTrip columns by value 0.
        dfHTS['nMinsTrip%d' % iTrip] = dfHTS['nMinsTrip%d' % iTrip].replace([-1], 0)

    dfHTS.to_csv('./tmpOutputs/dfHTS_tripProcessed1.csv', index=False)

    # gets rows that have at least 1 trip with unknown origin TZ or unknown destination TZ
    dfMissingTZs = dfHTS.loc[(dfHTS['tzOrigTrip1'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip1'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip2'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip2'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip3'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip3'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip4'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip4'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip5'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip5'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip6'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip6'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip7'].isin(missingTZs)) |
                             (dfHTS['tzDestTrip7'].isin(missingTZs)) |
                             (dfHTS['tzOrigTrip1'] > 265) |
                             (dfHTS['tzDestTrip1'] > 265) |
                             (dfHTS['tzOrigTrip2'] > 265) |
                             (dfHTS['tzDestTrip2'] > 265) |
                             (dfHTS['tzOrigTrip3'] > 265) |
                             (dfHTS['tzDestTrip3'] > 265) |
                             (dfHTS['tzOrigTrip4'] > 265) |
                             (dfHTS['tzDestTrip4'] > 265) |
                             (dfHTS['tzOrigTrip5'] > 265) |
                             (dfHTS['tzDestTrip5'] > 265) |
                             (dfHTS['tzOrigTrip6'] > 265) |
                             (dfHTS['tzDestTrip6'] > 265) |
                             (dfHTS['tzOrigTrip7'] > 265) |
                             (dfHTS['tzDestTrip7'] > 265)]
    # gets household ids corresponding to these rows of missing TZs
    uHhIDsMissingTZs = dfMissingTZs['hhID'].unique().tolist()
    # removes rows corresponding to these households from HTS
    dfHTS.drop(dfHTS[dfHTS['hhID'].isin(uHhIDsMissingTZs)].index, inplace=True)
    # drops original columns of origin and destination travel zones
    dfHTS.drop(columns=['tzOrigTrip1', 'tzDestTrip1', 'tzOrigTrip2', 'tzDestTrip2',
                        'tzOrigTrip3', 'tzDestTrip3', 'tzOrigTrip4', 'tzDestTrip4',
                        'tzOrigTrip5', 'tzDestTrip5', 'tzOrigTrip6', 'tzDestTrip6',
                        'tzOrigTrip7', 'tzDestTrip7'], inplace=True)

    dfHTS.to_csv('./tmpOutputs/dfHTS_tripProcessed2.csv')

    # if any of the trip detail is missing, set all other trip attribs to -1 for that trip. Apply to all trips.
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        selIndex = dfHTS.index[(dfHTS['typeOrigTrip%d' % iTrip].isna()) |
                               (dfHTS['typeDestTrip%d' % iTrip].isna()) |
                               (dfHTS['modeTrip%d' % iTrip].isna()) |
                               (dfHTS['purposeTrip%d' % iTrip].isna()) |
                               (dfHTS['nMinsTrip%d' % iTrip] == 0)]
        if len(selIndex) > 0:
            dfHTS['typeOrigTrip%d' % iTrip].loc[selIndex] = None
            dfHTS['typeDestTrip%d' % iTrip].loc[selIndex] = None
            dfHTS['modeTrip%d' % iTrip].loc[selIndex] = None
            dfHTS['purposeTrip%d' % iTrip].loc[selIndex] = None
            dfHTS['nMinsTrip%d' % iTrip].loc[selIndex] = 0
            dfHTS['nMinsDepTimeTrip%d' % iTrip].loc[selIndex] = 0
            dfHTS['origTrip%d' % iTrip].loc[selIndex] = None  # just to be sure, this should have been handled above
            dfHTS['destTrip%d' % iTrip].loc[selIndex] = None  # just to be sure, this should have been handled above

    dfHTS.to_csv('./tmpOutputs/dfHTS_tripProcessed3.csv', index=True)


# ======================================================================================================================
def oneHotEncodeAttribs(dfHTS):
    # one hot encodes trip attributes
    tripAttribs = ['origTrip', 'destTrip', 'typeOrigTrip', 'typeDestTrip', 'purposeTrip', 'modeTrip']
    for tripAttrib in tripAttribs:
        for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
            dummies = pd.get_dummies(dfHTS['%s%d' % (tripAttrib, iTrip)], prefix='%s%d' % (tripAttrib, iTrip))
            dfHTS = dfHTS.join(dummies)
            dfHTS.drop(columns=['%s%d' % (tripAttrib, iTrip)], inplace=True)

    # one hot encodes demographic attributes
    demoAttribs = ['gender', 'eduLevel', 'licence', 'jobType', 'employStatus', 'monthlyIncome', 'ownMotorVehicles']
    for attrib in demoAttribs:
        dummies = pd.get_dummies(dfHTS[attrib], prefix=attrib)
        dfHTS = dfHTS.join(dummies)
        dfHTS.drop(columns=[attrib], inplace=True)

    # drops redundant one-hot-encoded columns
    # dfHTS.drop(columns=['gender_female', 'eduLevel_NA', 'licence_no', 'jobType_other',
    #                    'ownMotorVehicles_no'], inplace=True)

    dfHTS.to_csv('./tmpOutputs/dfHTS_ohe0.csv', index=False)


# ======================================================================================================================
def getTripsCols(dfHTS):
    # makes a copy of dfHTS that has only trip attributes
    tripAttribTypes = ['origTrip', 'destTrip', 'typeOrigTrip', 'typeDestTrip',
                       'purposeTrip', 'modeTrip', 'nMinsTrip', 'nMinsDepTimeTrip']
    tripAttribCols = []
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        for attrib in tripAttribTypes:
            tripAttribCols.append('%s%d' % (attrib, iTrip))
    return dfHTS[tripAttribCols]


# ======================================================================================================================
def addIntraDistTripCol(dfTrips):
    # adds a column to indicate if a trip is intra-district
    def isIntraDistTrip(row, tripNum):
        if (row['origTrip%d' % tripNum] is None) or (row['destTrip%d' % tripNum] is None) or \
                (row['origTrip%d' % tripNum] != row['destTrip%d' % tripNum]):
            return 0
        else:
            return 1

    for iTrip in range(1, 7 + 1):
        dfTrips['intraDistTrip%d' % iTrip] = dfTrips.apply(lambda row: isIntraDistTrip(row, iTrip), axis=1)

# ======================================================================================================================
def removeOriginCols(dfTrips, origColTypes):
    origCols2Remove = []
    for colType in origColTypes:
        for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
            origCols2Remove.append('%s%d' % (colType, iTrip))
    dfTrips.drop(columns=origCols2Remove, inplace=True)

# ======================================================================================================================
def removeDestinationCols(dfTrips, destColTypes):
    destCols2Remove = []
    for colType in destColTypes:
        for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
            destCols2Remove.append('%s%d' % (colType, iTrip))
    dfTrips.drop(columns=destCols2Remove, inplace=True)

# ======================================================================================================================
def getLastTripId(currentRow):
    lastTrip = 1
    for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
        dest = currentRow['destTrip%d' % iTrip]
        if dest is None:
            lastTrip = iTrip - 1
            break
    return lastTrip


def getLastTripDetails(dfTrips):
    dfTrips['lastTrip'] = dfTrips.apply(lambda row: getLastTripId(row), axis=1)
    dfTrips['destLastTrip'] = dfTrips.apply(lambda row: row['destTrip%d' % row['lastTrip']], axis=1)
    dfTrips['typeDestLastTrip'] = dfTrips.apply(lambda row: row['typeDestTrip%d' % row['lastTrip']], axis=1)
    dfTrips.drop(columns=['lastTrip'], inplace=True)
    dfTrips['same1stOrigLastDest'] = dfTrips.apply(lambda row: 1 if row['destLastTrip'] == row['origTrip1'] else 0,
                                                   axis=1)

# ======================================================================================================================
def getNumOfTripsMade(dfTrips):
    dfTrips['nTrips'] = dfTrips.apply(lambda row: getLastTripId(row), axis=1)

# ======================================================================================================================
def oneHotEncodeTripAttribs(dfTrips, catAttribs):
    # one hot encodes categorical trip attributes. Note that 'origTrip' and 'typeOrigTrip' columns no longer exist.
    for tripAttrib in catAttribs:
        for iTrip in range(1, 7 + 1):  # for each trip from 1 to 7
            dummies = pd.get_dummies(dfTrips['%s%d' % (tripAttrib, iTrip)], prefix='%s%d' % (tripAttrib, iTrip))
            dfTrips = dfTrips.join(dummies)
            dfTrips.drop(columns=['%s%d' % (tripAttrib, iTrip)], inplace=True)
    # print(dfHTSTrips.columns)
    # print(len(dfHTSTrips.columns.tolist()))
    return dfTrips

# ======================================================================================================================
def standardise(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df
