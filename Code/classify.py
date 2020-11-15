import sys
import numpy as np
import datetime
import pandas as pd
# from numpy import genfromtxt



class Data:
    def __init__(self, filename):
        df = pd.read_csv(filename, delimiter=",")
        df.columns = ["original_index", "rss", "epc", "time", "diff"]
        df["time"].astype('int')

        t1 = df[df["epc"] == 9107]
        t2 = df[df["epc"] == 9108]
        t1 = t1.reset_index()
        t2 = t2.reset_index()

        startTime = t1["time"][0] if t1["time"][0] < t2["time"][0] else t2["time"][0]
        t1["time"] = t1["time"].apply(lambda x: x - startTime)
        t2["time"] = t2["time"].apply(lambda x: x - startTime)

        t1 = t1.drop(columns=["index", "epc"])
        t2 = t2.drop(columns=["index", "epc"])  

        self.t1 = np.array(t1["rss"])   #rss array for tag1
        self.t2 = np.array(t2["rss"])   #rss array for tag2

        #calculate baseline, mean value works fine
        self.base1 = np.mean(self.t1)   
        self.base2 = np.mean(self.t2)

        #calculate derivative of rss
        #can also use column "diff" in example 
        #data because they are preprocessed
        self.diff1 =  np.append(self.t1[1: ] - self.t1[: - 1], 0)
        self.diff2 =  np.append(self.t2[1: ] - self.t2[: - 1], 0)
        self.PI1, self.PI2 = self.findPosImpulse()
        self.NI1, self.NI2 = self.findNegImpulse()
        self.iM1, self.iM2 = self.findMaxIndex()
        self.getPeakRatio()





    def findPosImpulse(self):
        # m1 and m2 are indices of max values in rss derivatives
        # we can use them directly as postive impulse
        # but the actual impulse may end up being right to max value
        m1 = self.diff1.argmax()
        m2 = self.diff2.argmax()

        # go right until see decrease
        for i in range(m1, m1 + 3 if m1 + 3 < self.t1.shape[0] else self.t1.shape[0] ):
            if self.t1[i] >= self.t1[m1]:
                m1 = i
            else:
                break

        for i in range(m2, m2 + 3 if m2 + 3 < self.t2.shape[0] else self.t2.shape[0]):
            if self.t2[i] >= self.t2[m2]:
                m2 = i
            else:
                break
        return m1, m2

    def findNegImpulse(self):
        # m1 and m2 are indices
        m1 = self.diff1.argmin()
        m2 = self.diff2.argmin()
        # go left until see decrease
        for i in reversed(range(m1 - 3 if m1 - 3 >= 0 else 0, m1)):
            if self.t1[i] >= self.t1[m1]:
                m1 = i
            else:
                break

        for i in reversed(range(m2 - 3 if m2 - 3 >= 0 else 0, m2)):
            if self.t2[i] >= self.t2[m2]:
                m2 = i
            else:
                break

        return m1, m2

    def findMaxIndex(self):
        # the index of maximum rss value
        m1 = self.t1.argmax()
        m2 = self.t2.argmax() 
        return m1, m2

    def getPeakRatio(self):
        # ratio between peaks in diff
        self.Max1 = self.diff1.max()
        self.Max2 = self.diff2.max()

        self.Min1 = abs(self.diff1.min())
        self.Min2 = abs(self.diff2.min())

        self.R1 = self.Max1/ self.Min1 if self.Max1 >  self.Min1 else  self.Min1/self.Max1
        self.R2 = self.Max2/self.Min2 if self.Max2 > self.Min2 else self.Min2/self.Max2
   

class Classifier:
    def __init__(self, P, D):
        #hyper parameter 1, threshold of peak ratio
        #if peak ratio > this value
        #we decide that a peak is present
        self.P = P  
        #hyper parameter 2, threshold between max value and impulse
        #"offset" in paper
        self.D = D
    def classify(self, data):        
        classifyB = self.identifyB(data)
        if classifyB == None:
            return self.identifyACDE(data)
        else:
            return classifyB
    def identifyB(self, data):
        """
        STEP 1 in paper
        """
        # three possible results
        # 1. None: data is not B
        # 2. B1
        # 3. B2
        result = None
        # some values to be used
        """ 
        dd is the difference between impulses
        At B, for example: 
            positive impulse = [20, 21]
            negative impulse = [60, 62]
            dd = [40, 41]
        Two elements in dd have same sign, if two elements
        have different sign, it must not be B. Reverse is not true
        """
        dd1 = data.PI1 - data.NI1
        dd2 = data.PI2 - data.NI2


        """
        PValue and NValue
        They are the absolute Max and Min value of diff => "maximum and minimum peak" so to speak

        if the bigger value of them divide by smaller of them > certain threshold (P), then we say this
        tag only has 1 peak
        """
        
        tag1HasOnePeak = data.R1 > self.P

        
        tag2HasOnePeak = data.R2 > self.P
        """
        End of values to be used
        """

        # 1. check if impulse are at different sides
        if (dd1  * dd2 < 0):
            result = None
            # print("impulse different sides")
        # 2. check if PImpulse and NImpulse are near
        # parameter no. 1 : 5
        elif abs(dd1) < 5 or abs(dd2) < 5:
            result = None
            # print("impulse too near")

        # 3. check if impulse too close to origin
        # elif data.PI1 < 5 or data.PI2 < 5 or data.NI1 < 5 or data.NI2 < 5:
        #     result = None
        #     print("impulse too close to origin")

        # 5. check if tag1 impulses are imbalanced
        # 6. check if tag2 impulses are imbalanced
        if (tag1HasOnePeak and tag2HasOnePeak):
            # print("only one peak")

#         elif (tag1HasOnePeak or tag2HasOnePeak):
            result = None
        # 7. find window and trend of RSS within window
        else:
            d1 = data.t1[data.PI1: data.NI1]
            d2 = data.t2[data.PI2: data.NI2]
            if (len(d1) <= 1 or len(d2) <= 1 ):
                # print("window == 0")
                result = None
            else:
                #sign of coeff is the trend
                """
                STEP 4 in paper
                """
                coeff1 = np.polyfit(np.arange(0, len(d1)),d1,1)[0]
                coeff2 = np.polyfit(np.arange(0, len(d2)),d2,1)[0]
                if (coeff1 * coeff2 > 0):
                    # print("coeff sign same")
                    result = None
                elif (coeff1 > 0):
                    # result = "B2"
                    result = "CB"
                elif (coeff2 > 0):
                    # result = "B1"
                    result = "BC"
        return result

    def identifyACDE(self, data):
        """
        STEP 2 in paper
        """
        impulse1 = 0
        impulse2 = 0

        isPositiveImpulse = True
        impulseDiff = (data.Max1 - data.Min1) + (data.Max2 - data.Min2)
        
        if (impulseDiff >= 0):
            # Positive Impulse is larger
            # eg: 3, 4 - 2, 1 = 1, 3
            impulse1 = data.PI1
            impulse2 = data.PI2
        else:
            # Negative Impulse is larger
            # eg 2, 1 - 4, 3 = -2, -2
            impulse1 = data.NI1
            impulse2 = data.NI2
            isPositiveImpulse = False
        
        distanceBetweenMaxAndImpulse = np.array([abs(data.iM1 - impulse1), abs(data.iM2 - impulse2)])
        if (distanceBetweenMaxAndImpulse <= self.D).all():
            # max value and impulse are at roughly the same position
            # ab/ba, cd/dc
            """
            STEP 3 in paper
            """
            # check tag1 and 2 RSS relation at impulse
            d1 = data.t1[impulse1] -  data.t2[impulse2]
            # did not use relative RSS here to optimize 
            # for online classification
            
            pred = "C"
            if d1 > 0:
                pred = "A"
            if (pred == "A"):
                # we are at A
                if isPositiveImpulse:
                    return "BA"
                else:
                    return "AB"
            else:
                # we are at C
                if isPositiveImpulse:
                    return "CD"
                else:
                    return "DC"
        else:
            # abc/cba, bcd/dcb
            """
            STEP 3 in paper
            """
            # same as in paper, which tag has a bigger offset?
            if np.argmax(np.absolute(np.array([data.iM1 - impulse1, data.iM2 - impulse2]))) == 0:
                if isPositiveImpulse:
                    return "CBA"
                else: 
                    return "ABC"
            else:
                if isPositiveImpulse:
                    return "BCD"
                else:
                    return "DCB"

P = 1.8 #peak ratio, parameter 1
D = 50  #offset, hyperparameter 2
c = Classifier(P, D)

if len(sys.argv) <= 1:
    print("must provide data file path")
else:
    d = Data(sys.argv[1])
    print(c.classify(d))