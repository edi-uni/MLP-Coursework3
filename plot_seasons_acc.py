import matplotlib.pyplot as plt

seasons = {'1996-97_0': 0.3725, '1996-97_1': 0.5325454704165459, '1996-97_2': 0.7308000160455703, '1996-97_3': 0.6341818079352379, '1996-97_4': 0.5406000154912471, '1996-97_5': 0.5416666587591171, '1996-97_6': 0.722, '1996-97_7': 0.5360000176131725, '1996-97_8': 0.36260000759363176, '1996-97_9': 0.33000001311302185, '1997-98_0': 0.3650909199714661, '1997-98_1': 0.5084285728931427, '1997-98_2': 0.6305454545021058, '1997-98_3': 0.6648571501970291, '1997-98_4': 0.6033333550095559, '1997-98_5': 0.6683077120780945, '1997-98_6': 0.6760000193119049, '1997-98_7': 0.5904444494247436, '1997-98_8': 0.3333333432674408, '1997-98_9': 0.514375, '1998-99_0': 0.37475, '1998-99_1': 0.5355714412927628, '1998-99_2': 0.34540000614523886, '1998-99_3': 0.5289999932050705, '1998-99_4': 0.5355714413523674, '1998-99_5': 0.6087692460417747, '1998-99_6': 0.614375, '1998-99_7': 0.6685714450478554, '1998-99_8': 0.6225454617142677, '1998-99_9': 0.538875, '1999-00_0': 0.5534782484769821, '1999-00_1': 0.42200000500679014, '1999-00_2': 0.44825, '1999-00_3': 0.5926087124347686, '1999-00_4': 0.4114736784100533, '1999-00_5': 0.568125, '1999-00_6': 0.5490399982929229, '1999-00_7': 0.5256923233270645, '1999-00_8': 0.6154545480012894, '1999-00_9': 0.41582607969641683, '2000-01_0': 0.41582607969641683, '2000-01_1': 0.41582607969641683, '2000-01_2': 0.41582607969641683, '2000-01_3': 0.41582607969641683, '2000-01_4': 0.41582607969641683, '2000-01_5': 0.41582607969641683, '2000-01_6': 0.41582607969641683, '2000-01_7': 0.41582607969641683, '2000-01_8': 0.6417272675037384, '2000-01_9': 0.6370400121212005, '2001-02_0': 0.6370400121212005, '2001-02_1': 0.6370400121212005, '2001-02_2': 0.6370400121212005, '2001-02_3': 0.6370400121212005, '2001-02_4': 0.6370400121212005, '2001-02_5': 0.6370400121212005, '2001-02_6': 0.6370400121212005, '2001-02_7': 0.452, '2001-02_8': 0.5862941272258758, '2001-02_9': 0.6171875, '2002-03_0': 0.6171875, '2002-03_1': 0.6171875, '2002-03_2': 0.6171875, '2002-03_3': 0.6171875, '2002-03_4': 0.6171875, '2002-03_5': 0.6171875, '2002-03_6': 0.6171875, '2002-03_7': 0.6171875, '2002-03_8': 0.7292941325306892, '2002-03_9': 0.7396128984689713, '2003-04_0': 0.7396128984689713, '2003-04_1': 0.7396128984689713, '2003-04_2': 0.7396128984689713, '2003-04_3': 0.7396128984689713, '2003-04_4': 0.7396128984689713, '2003-04_5': 0.7396128984689713, '2003-04_6': 0.7396128984689713, '2003-04_7': 0.47657143020629883, '2003-04_8': 0.5018399928212166, '2003-04_9': 0.5607999964952469, '2004-05_0': 0.5607999964952469, '2004-05_1': 0.5607999964952469, '2004-05_2': 0.5607999964952469, '2004-05_3': 0.5607999964952469, '2004-05_4': 0.5607999964952469, '2004-05_5': 0.5607999964952469, '2004-05_6': 0.5607999964952469, '2004-05_7': 0.7491666668653488, '2004-05_8': 0.7243636569976807, '2004-05_9': 0.5383157913684845, '2005-06_0': 0.3938461573123932, '2005-06_1': 0.5941875, '2005-06_2': 0.5859999997019768, '2005-06_3': 0.6048800058066844, '2005-06_4': 0.7401290245056152, '2005-06_5': 0.395942864716053, '2005-06_6': 0.5188965475559235, '2005-06_7': 0.6858333320617676, '2005-06_8': 0.6489999985694885, '2005-06_9': 0.5102222247868776, '2006-07_0': 0.43521428340673446, '2006-07_1': 0.4761904776096344, '2006-07_2': 0.4509714390039444, '2006-07_3': 0.6074615440964699, '2006-07_4': 0.5659230842590331, '2006-07_5': 0.4994000011086464, '2006-07_6': 0.5252800001502037, '2006-07_7': 0.6311578698158264, '2006-07_8': 0.5179333472251892, '2006-07_9': 0.48104347294569016, '2007-08_0': 0.5239230822324753, '2007-08_1': 0.6763571543991566, '2007-08_2': 0.6019999985694885, '2007-08_3': 0.550620687007904, '2007-08_4': 0.626888890504837, '2007-08_5': 0.5788000013828277, '2007-08_6': 0.6980800153017044, '2007-08_7': 0.6574838753938675, '2007-08_8': 0.6652258004546165, '2007-08_9': 0.5730000110864639, '2008-09_0': 0.5115151591300965, '2008-09_1': 0.48951351302862167, '2008-09_2': 0.5358499900102616, '2008-09_3': 0.5007878841757775, '2008-09_4': 0.5877058909535408, '2008-09_5': 0.5649743832945824, '2008-09_6': 0.577724143922329, '2008-09_7': 0.4847407397478819, '2008-09_8': 0.455818195104599, '2008-09_9': 0.5357241306900978, '2009-10_0': 0.46814815640449525, '2009-10_1': 0.6111999990344048, '2009-10_2': 0.5086000058054924, '2009-10_3': 0.5610476360321045, '2009-10_4': 0.5473999997377396, '2009-10_5': 0.5233684212863445, '2009-10_6': 0.6023225733041764, '2009-10_7': 0.710722219824791, '2009-10_8': 0.4424799966812134, '2009-10_9': 0.6442777652740479, '2010-11_0': 0.525375, '2010-11_1': 0.5378399926424027, '2010-11_2': 0.536482766136527, '2010-11_3': 0.49356521493196487, '2010-11_4': 0.5368800002336502, '2010-11_5': 0.6261481471061706, '2010-11_6': 0.48407693284749986, '2010-11_7': 0.5382142854332924, '2010-11_8': 0.5349166684746742, '2010-11_9': 0.5188421220183372, '2011-12_0': 0.2636666723191738, '2011-12_1': 0.5184347714781761, '2011-12_2': 0.3365555652976036, '2011-12_3': 0.47957895743846896, '2011-12_4': 0.6117142881155014, '2011-12_5': 0.49309091132879257, '2011-12_6': 0.539375, '2011-12_7': 0.4477777841091156, '2011-12_8': 0.5288148235976696, '2011-12_9': 0.5555789468288421, '2012-13_0': 0.5505, '2012-13_1': 0.6308799998760224, '2012-13_2': 0.4920000030398369, '2012-13_3': 0.431999999165535, '2012-13_4': 0.6074117881059646, '2012-13_5': 0.5546666771173477, '2012-13_6': 0.7046956465244293, '2012-13_7': 0.6371999949216842, '2012-13_8': 0.7058400009870529, '2012-13_9': 0.5260769320726395, '2013-14_0': 0.55, '2013-14_1': 0.0, '2013-14_2': 0.6666666865348816, '2013-14_3': 0.6666666865348816, '2013-14_4': 0.5, '2013-14_5': 1.0, '2013-14_6': 1.0, '2013-14_7': 1.0, '2013-14_8': 1.0, '2013-14_9': 0.768, '2014-15_0': 0.6256363559961319, '2014-15_1': 0.6666666865348816, '2014-15_2': 0.5984000234603882, '2014-15_3': 0.5003077048063278, '2014-15_4': 0.533272743165493, '2014-15_5': 0.65375, '2014-15_6': 0.6492000062465668, '2014-15_7': 0.7784000067710877, '2014-15_8': 0.8792727483510971, '2014-15_9': 0.4086666724085808, '2015-16_0': 0.32706667557358743, '2015-16_1': 0.5850526258945465, '2015-16_2': 0.81225, '2015-16_3': 0.46040000131726266, '2015-16_4': 0.6666666865348816, '2015-16_5': 0.800000011920929, '2015-16_6': 0.708428586602211, '2015-16_7': 0.47842106205224993, '2015-16_8': 0.6356470772624015, '2015-16_9': 0.7032857275009156}

seasons_split = {}

count = 0
sum = 0

arr = []

for k,v in seasons.items():
    count += 1
    if count % 10 == 0:
        sum += v
        name = k[:7]
        seasons_split[name] = (sum/10)
        arr.append(sum/10)
        sum = 0
    else:
        sum += v


l = list(range(1,21))

print (l)
x = list(seasons_split.keys())

plt.plot(l, arr)
plt.xticks(l, x, rotation='vertical')
plt.show()