##For RPI
import serial
import time

import cv2
import numpy as np

##For RPI(有差了Arduino再解開註解)
#arduino=serial.Serial(port='/dev/ttyACM0', baudrate=9600,timeout=.1)

#import matplotlib.pylab as plt
'''
This code is now able to find pure 2 average lines&funciton & delete unused 註解&抓出兩線交點 &Logic(Flag to Arduino) 
&如果沒有偵測到任何line就顯示"Can't detect any line"然後把左右兩條線都設為垂直的左右邊線(且不取交點了)
&如果只偵測到一條線(ex: average left_line is none)，就把left_line設為最左方的邊線，照樣作交點判斷
&水平線已經另外排除掉，交點的計算是單純用水平線以外的線(左右兩條線)去用的
&置入影片&注意hough出來的lane的截距是y截距
&變成了SpeedPercentFlag的-100%~0~100%
&可以自由調整:濾色,blurKernal,CannyLeast,CanntTop的參數
&加了AverageInsecX,Y，每10筆Frame資料得出的intersection去計算交點(那個藍色點點)
Logic輸出來加了給RPI的東西
'''
#【讀取影片】
cap=cv2.VideoCapture('./GFencePhoto/GrassFenceV1.mp4') #讀取影片
#print("Start")
#cap=cv2.VideoCapture(1) #直接讀取鏡頭，在此寫上"鏡頭編號"，內建電腦鏡頭編號是0，如果外接了一個鏡頭，那外接鏡頭的編號是1

'''
#照片檔案跟定義
imgF1 = cv2.imread('./GFencePhoto/GrassFence5.jpg')
imgF1 = cv2.resize(imgF1,(0,0),fx=0.5,fy=0.5)
print(imgF1.shape)
#plt.imshow(imgF1)
#plt.show()
'''
#定義函式
def Empty(x):
    pass
#Settling
def FindcolorBar(imgNow): #TrakerBar for找到正確的刪顏色參數值，傳入一張RGB照片回傳回傳一張只剩下我們調過濾後的顏色的RGB照片跟遮罩imgMask
    imgNowHSV = cv2.cvtColor(imgNow,cv2.COLOR_BGR2HSV)
    #創建調整視窗
    cv2.namedWindow('TrackBar1') #創建一個視窗1st[視窗名稱]
    #cv2.resizeWindow('TrackBar1',640,320) #調整視窗大小
    cv2.createTrackbar('HueMin', 'TrackBar1', 0, 179, Empty)
    cv2.createTrackbar('HueMax','TrackBar1', 0, 179, Empty)
    cv2.createTrackbar('Saturation Min','TrackBar1', 0, 255, Empty)
    cv2.createTrackbar('Saturation Max','TrackBar1', 0, 255, Empty)
    cv2.createTrackbar('ValueMin','TrackBar1', 0, 255, Empty)
    cv2.createTrackbar('ValueMax','TrackBar1', 0, 255, Empty)

    cv2.createTrackbar('BlurKenal','TrackBar1', 1, 9, Empty)
    cv2.createTrackbar('CannyLeastThr','TrackBar1', 50, 200, Empty)
    cv2.createTrackbar('CannyTopThr','TrackBar1', 50, 350, Empty)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HueMax', 'TrackBar1', 179)
    cv2.setTrackbarPos('Saturation Max', 'TrackBar1', 255)
    cv2.setTrackbarPos('ValueMax', 'TrackBar1', 255)

    cv2.setTrackbarPos('BlurKenal', 'TrackBar1', 5)
    cv2.setTrackbarPos('CannyLeastThr', 'TrackBar1', 50)
    cv2.setTrackbarPos('CannyTopThr', 'TrackBar1', 250)

    '''
    # Initialize HSV min/max values
    h_min = s_min = v_min = h_max = s_max = v_max = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    '''
    #cv2.waitKey(ord('D')) #當調整完畢就按下D
    while (1): 
        h_min = cv2.getTrackbarPos('HueMin', 'TrackBar1')
        h_max = cv2.getTrackbarPos('HueMax', 'TrackBar1')
        s_min = cv2.getTrackbarPos('Saturation Min','TrackBar1')
        s_max = cv2.getTrackbarPos('Saturation Max','TrackBar1')
        v_min = cv2.getTrackbarPos('ValueMin','TrackBar1')
        v_max = cv2.getTrackbarPos('ValueMax','TrackBar1')

        canny_least = cv2.getTrackbarPos('CannyLeastThr','TrackBar1')
        canny_top = cv2.getTrackbarPos('CannyTopThr','TrackBar1') 
        blur_kenal_Bar = cv2.getTrackbarPos('BlurKenal','TrackBar1')
        '''
        match blur_kenal_Bar :
            case 1:
                blur_kenal=(1,1)
            case 2:
                blur_kenal=(3,3)
            case 3:
                blur_kenal=(5,5)
            case 4:
                blur_kenal=(7,7)
            case 5:
                blur_kenal=(11,11)
            case 6:
                blur_kenal=(13,13)
            case 7:
                blur_kenal=(17,17)
            case 8:
                blur_kenal=(19,19)
            case 9:
                blur_kenal=(23,23)
        '''
        if blur_kenal_Bar==1:
            blur_kenal=(1,1)
        elif blur_kenal_Bar==2:
            blur_kenal=(3,3)
        elif blur_kenal_Bar==3:
                blur_kenal=(5,5)
        elif blur_kenal_Bar==4:
            blur_kenal=(7,7)
        elif blur_kenal_Bar==5:
            blur_kenal=(11,11)
        elif blur_kenal_Bar==6:
            blur_kenal=(13,13)
        elif blur_kenal_Bar==7:
            blur_kenal=(17,17)
        elif blur_kenal_Bar==8:
            blur_kenal=(19,19)
        elif blur_kenal_Bar==9:
            blur_kenal=(23,23)
        #print(h_min, h_max, s_min, s_max, v_min, v_max)
        #過濾顏色
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])    
        imgMask = cv2.inRange(imgNowHSV,lower,upper)#黑:被過濾掉的顏色區，白: 留下之區
        imgColorFilt = cv2.bitwise_and(imgNow, imgNow, mask=imgMask)
        
        #blur_kenal=(13,13)
        gray = cv2.cvtColor(imgColorFilt, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_kenal, 0)
        cv2.imshow('GrayBlur',gray)
        
        imgCanny=FindCanny(gray,canny_least,canny_top)
        
        cv2.imshow('Canny Photo',imgCanny)
        '''
        #試Canny
        blur_kenal=(13,13)
        gray = cv2.cvtColor(imgColorFilt, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_kenal, 0)
        imgCanny=FindCanny(gray,canny_least,canny_top)
        cv2.imshow=('Canny Photo',imgCanny)
        '''
        #cv2.imshow('Filtered Image',imgColorFilt)
        cv2.imshow('TrackBar1',imgColorFilt) #可以圖片也加在這個trackbar視窗上，這樣都整合在一起了
       
        #要加if,要有break!!不然電腦判斷自己卡在迴圈裡就當機了
        if cv2.waitKey(10) & 0xFF == ord('d'):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (h_min , s_min , v_min, h_max, s_max , v_max))
            cv2.destroyAllWindows()
            break
    
    return h_min,h_max,s_min,s_max,v_min,v_max,canny_least,canny_top,blur_kenal
    #return imgColorFilt,imgMask  #回傳一張只剩下我們過濾後的顏色的照片

#def OriginalSettings()

def PureColorfilter(h_min,h_max,s_min,s_max,v_min,v_max,imgFrameNow):
    imgNowHSV = cv2.cvtColor(imgFrameNow,cv2.COLOR_BGR2HSV)
    #過濾顏色
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])    
    imgMask = cv2.inRange(imgNowHSV,lower,upper)#黑:被過濾掉的顏色區，白: 留下之區
    imgColorFilt = cv2.bitwise_and(imgF1, imgF1, mask=imgMask)
    cv2.imshow('imgMask',imgMask)
    return imgColorFilt,imgMask
            


def FindCanny(FiltedImgNow,canny_least=50,canny_top=200):#傳入一張已經過濾過顏色的RGB照片(或是Mask)，回傳照片的邊緣
    #<先找邊緣>
    imgCanny = cv2.Canny(FiltedImgNow, canny_least, canny_top)  #先找邊緣canny
    #cv2.imshow('j',imgCanny)
    #cv2.waitKey(0)
    #<再找輪廓>:輪廓點儲存在contours變數
    #回傳輪廓點陣列/階層       #找輪廓                  偵測外輪廓        近似方法:no(即保留所有輪廓點)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    return imgCanny

#用HoughLinesP(傳出x,y有效率)
def FindHoughLine(CannyImgNow):#傳入一個輪廓img(CannyImg)，傳出那幾個有線的方程式，回傳average line(left_line,right_line)並且畫出所有line跟averageline
    copyImgF1=imgF1.copy()
    lines = cv2.HoughLinesP(CannyImgNow, rho=1, theta=np.pi/180,  threshold=100, minLineLength=0, maxLineGap=50)#可調threshold(門檻)/minLineLength=50/maxLineGap=30
    if lines is not None:
        draw_houghLines(copyImgF1, lines, [255, 0, 0], 2)#傳入hough產出的多條線
        left_line,right_line,parallel_line=lane_lines(copyImgF1, lines)#回傳平均後的左右兩條線兩點資料(left_"line",right_"line")
        #print(left_line,right_line)
        if left_line is not None:
            draw_averLines(copyImgF1, left_line, [0, 255, 0], 3)#draw_lines(image, lines, color=[0, 255, 0], thickness=3):
        if right_line is not None:
            draw_averLines(copyImgF1, right_line, [0, 255, 0], 3)
        if parallel_line is not None:
            draw_averLines(copyImgF1, parallel_line, [0, 255, 0], 3)

        InSecX,InSecY,copyImgF1=lineIntersection(left_line,right_line,copyImgF1)
        #LogicDetermed(InSecX,InSecY,imgF1)
    else :
        print("Can't detect any line")
        left_line=[[0,0],[0,copyImgF1.shape[0]]]
        right_line=[[copyImgF1.shape[1],0],[copyImgF1.shape[1],copyImgF1.shape[0]]]
        InSecX=None
        InSecY=None

        
    cv2.imshow('Detecting Line Image',copyImgF1)
    cv2.imshow('Canny Edges Now',CannyImgNow)
    #cv2.waitKey(0)
    '''
    print("Now Left line2=" )
    print(left_line)
    print("Now right line2=")
    print(right_line)
    '''
    #cv2.circle(copyImgF1, (361,173), radius=100, color=[0,255,0], thickness=-1)
    return left_line,right_line,copyImgF1,InSecX,InSecY#此為average line(兩線有各兩點x1,y1/x2,y2)
    

############找到平均的線條#############
#參考資料: https://github.com/mohamedameen93/Lane-lines-detection-using-Python-and-OpenCV/blob/master/Writeup.md
def average_slope_intercept(lines,image):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    parallel_lines = []
    parallel_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if -0.16<slope<0.16:#如果斜率為+-0.16(即斜率是+-9.09)判斷為水平線
                parallel_lines.append((slope, intercept))
                parallel_weights.append((length))
            elif slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    if len(left_weights) > 0:            
        left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights) 
        #print("Now left_lane")
        #print(left_lane)
    else :
        left_lane=[-100,0]#[[0,0],[0,image.shape[0]]]#如果只偵測到一條線(即left_line是none)的話，會出問題，所以在此設定如果left_line是none，就把left_line設為最左邊的那條線，這樣就還是能算出交點且邏輯不變
    if len(right_weights) > 0 :
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) 
    else :
        right_lane=[100,-100*image.shape[1]]#[[image.shape[1],0],[image.shape[1],image.shape[0]]]
    #left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    #right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    if len(parallel_weights) > 0 :
        parallel_lane = np.dot(parallel_weights, parallel_lines) / np.sum(parallel_weights) 
    else :
        parallel_lane=None
    return left_lane, right_lane,parallel_lane #回傳的是剩下slope跟一經過點(x1,y1)資料的兩條average線("lane")

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    
    if -0.16<slope<0.16:#預防slope太小近0，(y1 - intercept)/slope會變成分母是0，所以slope太小就直接訂為臨界值的+-0.05
        
        x1=0
        x2=imgF1.shape[1]
        y1=int(intercept)#intercept is y intercept
        y2=y1
        #if slope>0:
        #if slope<0:
        #    slope=-1
        return [[x1, y1],[x2, y2]]
        
    
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    #return ((x1, y1), (x2, y2))
    return [[x1, y1],[x2, y2]]


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane,parallel_lane = average_slope_intercept(lines,image)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    if left_lane is not None:
        left_line  = pixel_points(y1, y2, left_lane)
    else: 
        left_line=[[0,0],[0,image.shape[0]]]

    if right_lane is not None:
        right_line = pixel_points(y1, y2, right_lane)
    else:
        right_line=[[image.shape[1],0],[image.shape[1],image.shape[0]]]

    if parallel_lane is not None:
        parallel_line = pixel_points(y1, y2, parallel_lane)
        #print("Parallel_Lane=")
        #yintercept=-parallel_lane[0]*parallel_lane[1]
        #print(yintercept)
        #不能用上面的，因為當slope近0時，x1 = int((y1 - intercept)/slope)，(y1 - intercept)/slope會變成無限大，沒辦法轉成int
        #parallel_line = [[0,parallel_lane[1]],[image.shape[1],parallel_lane[1]]]#直接近似成水平線(parallel_lane[1]即為y截距)
        #parallel_line = [[0,yintercept],[image.shape[1],yintercept]]
        #parallel_line = [[0,y1],[image.shape[1],y1]]
    else:
        parallel_line=None
    return left_line, right_line,parallel_line #回傳的是average後的左右兩條線，分別各線都有自己的(x1,y1),(x2,y2)兩點("line")

def draw_averLines(image, lines, color=[0, 255, 0], thickness=3):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.(你要畫在的照片上)
            lines: The output lines from Hough Transform.(你要畫的那條線)(務必是以矩陣形式處存過線的兩點[[x1,y1],[x2,y2]])ex:[[13, 393], [266, 235]]
            color (Default = green): Line color.
            thickness (Default = 3): Line thickness. 
    """
    x1,y1=lines[0]
    x2,y2=lines[1]
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_houghLines(image, lines, color=[0, 255, 0], thickness=3):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.(你要畫在的照片上)
            lines: The output lines from Hough Transform.(你要畫的那條線)(務必是以矩陣形式處存過線的兩點且hough出來的很多條線矩陣形式[[x1 y1 x2 y2]])ex:[[223 191 583 391]][[225 188 548 360]]
            color (Default = green): Line color.
            thickness (Default = 3): Line thickness. 
    """
    if lines is not None:
        for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

#################################
def lineIntersection(line_a,line_b,image):#輸入兩條線line_a,line_b和想要把點畫在的那個圖image，(找到兩直線交點並劃出點在image上)，並回傳交點x,y值InSecX,InSecY,跟那個畫上了點的圖
    #print("ok0")
    #print(lineA)
    ax1,ay1=line_a[0]
    ax2,ay2=line_a[1]
    bx1,by1=line_b[0]
    bx2,by2=line_b[1]
    '''
    print("NowLine")
    print(ax1,ay1,ax2,ay2)
    print("B")
    print(bx1,by1,bx2,by2)
    '''
    MatrixC=np.array([[(ay2-ay1), -(ax2-ax1)],[(by2-by1),-(bx2-bx1)]])
    MatrixD=np.array([[ax1*(ay2-ay1)-ay1*(ax2-ax1)],[bx1*(by2-by1)-by1*(bx2-bx1)]])

    IntersectionXY=np.linalg.solve(MatrixC,MatrixD)
    InSecX=int(IntersectionXY[0][0])
    InSecY=int(IntersectionXY[1][0])
    print("Now intersection=(%d,%d)"%(InSecX,InSecY))
    #dotCenter=(InSecX,InSecY)
                     #因為是pixcel 所以圓中心要是整數(沒有半個pixcel這種的)
    cv2.circle(image,(InSecX,InSecY) , radius=5, color=(0,0,255), thickness=-1)
    #cv2.imshow('dotImg',image)
    return InSecX,InSecY,image

'''
主要運行的function:
:每一個frame的影像處理抓出焦點之類的
LogicDetermed():邏輯判斷回傳值
'''

def LogicDetermed(inSecX,inSecY,imgF1):
    ImgShape=imgF1.shape
    print(ImgShape)
    ImgShapeY=ImgShape[0]#圓圖的Y座標
    ImgShapeX=ImgShape[1]

    #cv2.imshow('Logic Determinator',imgF1)
    #print(ImgShapeX)
    ######start logic#####
    #ps: 下面的intersectionX,Y都是左右兩條邊線的(已排除所有水平線所作的交點)
    
    if inSecX < ImgShapeX/2:
        print("turn left")
        #要傳給arduino的就是DirectionFlag(char)跟SpeedFlag(0-255)
        #DirectionFlag='L'
        #SpeedPercentFlag=100#為一0~100的數字，用來給arduino要轉多快

        
    elif (ImgShapeX/2)<inSecX:
        print("turn right")
        #DirectionFlag='R'
        #SpeedFlag='100'

    
    SpeedPercentFlag=((inSecX-ImgShapeX/2)/(ImgShapeX/2))*100
    print('NowPFlag=%d'%SpeedPercentFlag)

    return SpeedPercentFlag


'''
def hough_inter(theta1, rho1, theta2, rho2):
    A = np.array([[np.cos(theta1), np.sin(theta1)], 
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.lstsq(A, b)[0] # use lstsq to solve Ax = b, not inv() which is unstable
'''
#############正式的程式main############
#imgF1為影像處理後的Frame
ColorBarFlag=0#讓FindcolorBar只有第一次被做而已

SumInSecX=0
SumInSecY=0
SumInSecCount=-1#-1專屬用來給第一群組average的初始值
AverageInSecX=0
AverageInSecY=0

while True:
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (600,300))
    #frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)
    imgF1 = frame.copy()
    if ColorBarFlag==0: #Parameters Settling
        h_min,h_max,s_min,s_max,v_min,v_max,canny_least,canny_top,blur_kenal=FindcolorBar(frame)
        ColorBarFlag=1

    
    if ret:

        cv2.imshow('video', frame)
        #播放影像處理後的Frame(imgF1)
        imgF1CFilted,imgMask = PureColorfilter(h_min,h_max,s_min,s_max,v_min,v_max,imgF1)
        gray = cv2.cvtColor(imgF1CFilted, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_kenal, 0)
        ImgCanny1=FindCanny(gray,canny_least,canny_top)
        left_line, right_line,copyimgF1,NowInSecX,NowInSecY=FindHoughLine(ImgCanny1)
        ###Get Average InSecX,Y

        #NewAverageInSecX=SumInSecX/SumInSecCount
        #NewAverageInSecY=SumInSecY/SumInSecCount

        if SumInSecCount<10:#每100次frame取一次平均去修正
            if NowInSecX is not None and NowInSecY is not None:
                SumInSecX=SumInSecX+NowInSecX
                SumInSecY=SumInSecY+NowInSecY
                if SumInSecCount==-1:#給第一群組的Average初始值
                    AverageInSecX=int(NowInSecX)
                    AverageInSecY=int(NowInSecY)
                    SumInSecCount=0
                SumInSecCount=SumInSecCount+1
            #AverageInSecX=SumInSecX/SumInSecCount
            #AverageInSecY=SumInSecY/SumInSecCount
        else:
            AverageInSecX=int(SumInSecX/SumInSecCount)
            AverageInSecY=int(SumInSecY/SumInSecCount)
            SumInSecX=0
            SumInSecY=0
            SumInSecCount=0
        print("Average Intercept=(%d,%d)"%(AverageInSecX,AverageInSecY))
        cv2.circle(copyimgF1,(AverageInSecX,AverageInSecY) , radius=5, color=(255,255,0), thickness=-1)
        cv2.imshow('AverageInterCept',copyimgF1)
        SpeedPercentFlag=LogicDetermed(AverageInSecX,AverageInSecY,imgF1)
        
        ###用來傳入Arduino###(有插入Arduino再解開註解)
        #arduino.write(bytes(SpeedPercentFlag,'utf-8'))







    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break








    '''
    imgF1CFilted,imgMask = FindcolorBar(imgF1)

    #cv2.imshow('kk',imgF1CFilted)
    #cv2.imshow('mm',imgMask)

    gray = cv2.cvtColor(imgF1CFilted, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13,13), 0)
    #gray3 = cv2.GaussianBlur(imgMask, (11,11), 0)

    ImgCanny1=FindCanny(gray)
    #ImgCanny3=FindCanny(gray3)
    #FindHoughLine(ImgCanny1)
    #FindHoughLine(ImgCanny3)
    left_line, right_line,copyimgF1=FindHoughLine(ImgCanny1)


    #InSecX,InSecY,copyimgF1=lineIntersection(left_line,right_line,copyimgF1)
    #LogicDetermed(InSecX,InSecY,imgF1)

    cv2.waitKey(0)& 0xFF == ord('a')
    print("code End")
    '''