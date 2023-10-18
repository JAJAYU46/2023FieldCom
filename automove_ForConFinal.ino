#include <Ultrasonic.h>
#include <math.h>

//最終用來比賽的RPIwalkcod，把參數都變成變數了

//【走路的調整參數】
/*
明天要調整的東西:
1. 最大左轉參數:
NormalT(250, 0, 0, 250);
NormalT(200, 0, 0, 200);
NormalT(100, 0, 0, 250);
NormalT(100, 0, 0, 200);
上面是不了要是正轉
>>用於大左轉參數, 

2. 相機的大左轉RPI參數距離(配合大左轉參，可能要遠疫點早一點就開始大左轉):
試1/2,5/12,4/12,

3. 超音波f1,f2, 要多少開始退要退回多少try:
退20 
有左右轉就是用最大左右轉

4. left,right turn用最大左右轉值

5. 線性的調整的逆轉最大值(才不會跑不動):
測*200,*150,*100
(另一測配合最大左轉)

6. 看會不會卡石頭避障(可以多寫個兩次後退後看哪個front先測到), 如果柏油路卡了就寫個


7. 直走速度(影響超音波): (250,250)&(200,200)
*/

//Flag調整
int RPIWalkFlag = 1;
int WaterFlag = 0;
int ultFlag = 1;              //1要避障
double TurnFPer = 230 / 150;  //150
double TurnBPer = 1;          //150
double TurnRPer = 230 / 200;  //想要右轉照比例條多快多慢//200
double TurnLPer = 230 / 200;  //200

// 定義左右馬達的控制引腳
const int leftMotorPin1 = 12;   //左正
const int leftMotorPin2 = 13;   //左負
const int rightMotorPin1 = 10;  //右負
const int rightMotorPin2 = 11;  //右正
String movement;
int x;
int startTime;
int TurnFlag = 0;
int Duration;
double RMTonePara = 0.6;  //(調整左右馬達速差的參，乘在右邊馬達數值上)

//蔽障角位

Ultrasonic ultrasonic_front1(4, 5);
Ultrasonic ultrasonic_left(2, 3);
Ultrasonic ultrasonic_right(8, 9);
Ultrasonic ultrasonic_front2(6, 7);
//避障
int distance_front1, distance_front2, distance_left, distance_right;

//噴水角位
const int WaterPinL = 50;  //左邊的噴水腳位
const int WaterPinR = 51;  //右邊的噴水腳位

//避障超音波
void TurnRight(int lm1 = 0, int lm2 = 200, int rm1 = 200, int rm2 = 0) {  //leftMotor(1是逆轉)
  int NowTurnPer = TurnRPer;

  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
}

void TurnLeft(int lm1 = 200, int lm2 = 0, int rm1 = 0, int rm2 = 200) {  //leftMotor(1是逆轉)
  int NowTurnPer = TurnLPer;

  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
}

void Backward(int lm1 = 150, int lm2 = 0, int rm1 = 150, int rm2 = 0) {  //leftMotor(1是逆轉)
  int NowTurnPer = TurnBPer;

  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
}

void Frontward(int lm1 = 0, int lm2 = 150, int rm1 = 0, int rm2 = 150) {
  int NowTurnPer = TurnFPer;

  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
}

void Stop(int lm1 = 0, int lm2 = 0, int rm1 = 0, int rm2 = 0) {
  int NowTurnPer = 1;
  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
  delay(100);
}

void NormalT(int lm1 = 0, int lm2 = 0, int rm1 = 0, int rm2 = 0) {
  int NowTurnPer = 1;
  analogWrite(leftMotorPin1, int(lm1 * NowTurnPer));
  analogWrite(leftMotorPin2, int(lm2 * NowTurnPer));
  analogWrite(rightMotorPin1, int(rm1 * NowTurnPer * RMTonePara));
  analogWrite(rightMotorPin2, int(rm2 * NowTurnPer * RMTonePara));
  delay(100);
}
void ReadUlst() {
  distance_front1 = ultrasonic_front1.read();
  distance_left = ultrasonic_left.read();
  distance_right = ultrasonic_right.read();
  distance_front2 = ultrasonic_front2.read();
}

void setup() {
  // 將馬達控制引腳設置為輸出
  pinMode(leftMotorPin1, OUTPUT);
  pinMode(leftMotorPin2, OUTPUT);
  pinMode(rightMotorPin1, OUTPUT);
  pinMode(rightMotorPin2, OUTPUT);
  pinMode(WaterPinL, OUTPUT);
  pinMode(WaterPinR, OUTPUT);
  Serial.begin(115200);

  Stop();
  delay(1000);

  //while (true) {
  //NormalT(0,255,0,255);}
}

//別寫進loop fuction裡，不然出迴圈記憶體又被釋放了

int WaterState = 3000;
int WaterStateL = 0;
int WaterStateR = 0;
int WaterTimeFlagR = 0;
int WaterTimeFlagL = 0;
int startTimeL, startTimeR, DurationWL, DurationWR;  
//int startTimeL, startTimeR, DurationWL, DurationWR;

void loop() {
  int MotorSL2;
  int MotorSR2;




  //走路開關
  if (RPIWalkFlag == 1) {
    Duration = millis() - startTime;

    if (WaterFlag = 0) {
      if (Serial.available() > 0) {
        movement = Serial.readStringUntil('\n');
        x = movement.toInt();
      }
    } else if (WaterFlag = 1) {
      // Read the data sent from Python
      String data = Serial.readStringUntil('\n');

      // Split the data into two variables
      movement = data.substring(0, data.indexOf(','));
      x = movement.toInt();
      movement = data.substring(data.indexOf(',') + 1);
      WaterState = movement.toInt();

      if (WaterState == 3000) {
        WaterStateL = 0;
        WaterStateR = 0;
      } else if (WaterState == -2000) {
        WaterStateL = 1;
        WaterStateR = 0;
      } else if (WaterState == 2000) {
        WaterStateL = 0;
        WaterStateR = 1;
      } else if (WaterState == 2002) {
        WaterStateL = 1;
        WaterStateR = 1;
      }
    }



    if (x == 10000) {

      Stop();
    }
    if (x == -1000) {
      //當第一瞬間進入超大左轉迴圈

      if (TurnFlag == 0) {
        startTime = millis();
        TurnFlag = 1;
      }
      Stop();
      delay(500);
      Duration = millis() - startTime;
      if (Duration >= 0) {  //如果距離第一次大左轉信號已經過了300毫秒，就開始大左轉直到看到左右兩條直線(即進到arduino的其他迴，就把TurnFlag=0) //現在是把轉彎延遲功能關掉

        while (true) {
          NormalT(250, 0, 0, 250);  //250,0,0,250
          if (Serial.available() > 0) {
            movement = Serial.readStringUntil('\n');
            x = movement.toInt();
          }
          if (-100 <= x && x <= 100) {  //即RPI判斷它已經不再大左轉的範圍了(即RPI判斷看到兩條直線)
            TurnFlag = 0;
            break;
          }
        }
      }

    } else if (x == 1000) {
      //當第一瞬間進入超大左轉迴圈

      if (TurnFlag == 0) {
        startTime = millis();
        TurnFlag = 1;
      }
      Stop();
      delay(500);
      Duration = millis() - startTime;
      if (Duration >= 0) {  //如果距離第一次大左轉信號已經過了300毫秒，就開始大左轉直到看到左右兩條直線(即進到arduino的其他迴，就把TurnFlag=0) //現在是把轉彎延遲功能關掉


        while (true) {

          NormalT(0, 250, 240, 0);  //0,250,240,0
          if (Serial.available() > 0) {
            movement = Serial.readStringUntil('\n');
            x = movement.toInt();
          }
          if (-100 <= x && x <= 100) {  //即RPI判斷它已經不再大左轉的範圍了(即RPI判斷看到兩條直線)
            TurnFlag = 0;
            break;
          }
        }
      }
    }


    if (-20 < x && x < 20) {
      Frontward();
    } else if (-20 >= x && x >= -70) {     //如果在此區間就用線性調整
      MotorSL2 = int(abs(x) / 100 * 200);  //(當x是-100的時候大左轉MotorSL2=0)//*250//150
      TurnLeft(MotorSL2, 0, 0, 250);

    } else if (70 >= x && x >= 20) {  //如果在此區間就用線性調整

      MotorSR2 = int(abs(x) / 100 * 200);  //*220//250//150/////////////////////////////////////////////////////////////////
      TurnRight(0, 250, MotorSR2, 0);

    } else if (-100 <= x && x < -70) {

      TurnLeft(200, 0, 0, 250);  //200,0,0,200///150/////////////////////////////////////////////////////////////////////////////

    } else if (100 >= x && x > 70) {
      TurnRight(0, 250, 200, 0);  //0,200,200,0///150
    }
  }

  //避障功能

  if (ultFlag == 1) {

    ReadUlst();

    //直走靠opencv
    if (distance_front1 < 5 && distance_front2 < 5 && distance_front2 > 0 && distance_front1 > 0) {
      while (distance_front1 <= 10 || distance_front2 <= 10) {
        Backward();
        ReadUlst();
      }

    } else if (distance_front1 < 5 && distance_front1 > 0) {
      if (distance_left < 5) {
        Stop();
        while (distance_front1 <= 15) {
          Backward();
          ReadUlst();
        }
        while (distance_left <= 20) {
          TurnRight();
          ReadUlst();
        }


      } else {
        Stop();
        while (distance_front1 <= 15) {
          Backward();
          ReadUlst();
        }
      }
    } else if (distance_front2 < 5 && distance_front2 > 0) {  //&&distance_front2>0排除接線不良的0
      if (distance_right < 5) {
        Stop();
        while (distance_front2 <= 15) {
          Backward();
          ReadUlst();
        }
        while (distance_left <= 20) {
          TurnLeft();
          ReadUlst();
        }


      } else {
        Stop();
        while (distance_front1 <= 15) {
          Backward();
          ReadUlst();
        }
      }
    } else if (distance_left < 5 && distance_left > 0) {
      while (distance_left < 20) {
        TurnRight();
        ReadUlst();
      }
    } else if (distance_right < 5 && distance_right > 0) {
      while (distance_right < 20) {
        TurnLeft();
        ReadUlst();
      }
    }

    /*
    Serial.print("\ndistance_front1:");
    Serial.print(distance_front1);
    Serial.print("\ndistance_front2:");
    Serial.print(distance_front2);
    Serial.print("\ndistance_left:");
    Serial.print(distance_left);
    Serial.print("\ndistance_right:");
    Serial.print(distance_right);*/
  }
  //噴水功能



  DurationWL = millis() - startTimeL;
  DurationWR = millis() - startTimeR;

  //Serial.print("\nStartTimeL:");
  //Serial.print(startTimeL);
  //Serial.print("\nDurationWL:");
  //Serial.print(DurationWL);
  if (DurationWL >= 3000 && WaterTimeFlagL == 1) {
    digitalWrite(WaterPinL, LOW);
    WaterTimeFlagL = 0;
    //Serial.print("\nOver");
    delay(1000);
  }
  if (DurationWR >= 3000 && WaterTimeFlagR == 1) {
    digitalWrite(WaterPinR, LOW);
    WaterTimeFlagR = 0;
  }

  if (WaterStateL == 1) {  //左噴兩秒
    //Serial.print("ok1");
    //Serial.print("WaterTimeFlagL");
    //Serial.print(WaterTimeFlagL);
    //Serial.print("\n");
    if (WaterTimeFlagL == 0) {
      startTimeL = millis();
      //Serial.print("\nStartTimeL2:");
      //Serial.print(startTimeL);
      digitalWrite(WaterPinL, HIGH);

      /*Stop();
      delay(1000);
      Backward();
      delay(1000);
      Stop();
      delay(1000);*/
      WaterTimeFlagL = 1;
      //Serial.print("\nWaterTimeFlagL");
      //Serial.print(WaterTimeFlagL);
    }
  }
  if (WaterStateR == 1) {  //右噴兩秒
    if (WaterTimeFlagR == 0) {
      startTimeR = millis();
      digitalWrite(WaterPinR, HIGH);
      /*
      Stop();
      delay(1000);
      Backward();
      delay(1000);
      Stop();
      delay(1000);*/
      WaterTimeFlagR = 1;
    }
  }
  //Serial.print(WaterTimeFlagL);
}

/*
if (TurnFlag == 0) {
        startTime = millis();
        TurnFlag = 1;
      }
      Stop();
      delay(500);
      Duration = millis() - startTime;
      if (Duration >= 0) {  //如果距離第一次大左轉信號已經過了300毫秒，就開始大左轉直到看到左右兩條直線(即進到arduino的其他迴，就把TurnFlag=0) //現在是把轉彎延遲功能關掉

        while (true) {
          NormalT(250, 0, 0, 250);  //250,0,0,250
          if (Serial.available() > 0) {
            movement = Serial.readStringUntil('\n');
            x = movement.toInt();
          }
          if (-100 <= x && x <= 100) {  //即RPI判斷它已經不再大左轉的範圍了(即RPI判斷看到兩條直線)
            TurnFlag = 0;
            break;
          }
        }
      }
*/
//非線性調整
/*
  if (x >= -100 && x < -80) {
    analogWrite(leftMotorPin1, 200);
    analogWrite(leftMotorPin2, 0);
    analogWrite(rightMotorPin1, 0);
    analogWrite(rightMotorPin2, 220);


    // Serial.print("ok1");
  } else if (x >= -80 && x < -50) {
    analogWrite(leftMotorPin1, 100);
    analogWrite(leftMotorPin2, 0);  //100
    analogWrite(rightMotorPin1, 0);
    analogWrite(rightMotorPin2, 220);


    // Serial.print("ok2");
  } 
  else if (x >= -50 && x < 0) {
    analogWrite(leftMotorPin1, 0);
    analogWrite(leftMotorPin2, 220);  //100
    analogWrite(rightMotorPin1, 0);
    analogWrite(rightMotorPin2, 220);
    // Serial.print("ok2");
  } else if (x == 0) {
    analogWrite(leftMotorPin1, 0);
    analogWrite(leftMotorPin2, 220);
    analogWrite(rightMotorPin1, 0);
    analogWrite(rightMotorPin2, 220);
    //Serial.print("ok3");
  } else if (x > 0 && x <= 50) {
    analogWrite(leftMotorPin1, 0);
    analogWrite(leftMotorPin2, 220);
    analogWrite(rightMotorPin1, 0);
    analogWrite(rightMotorPin2, 220);  //100
    //Serial.print("ok4");
  }
  else if (x > 50 && x < 80) {
    analogWrite(leftMotorPin1, 0);
    analogWrite(leftMotorPin2, 220);
    analogWrite(rightMotorPin1, 100);
    analogWrite(rightMotorPin2, 0);  //100
                                     // Serial.print("ok2");
  } else if (x > 80 && x <= 100) {
    analogWrite(leftMotorPin1, 0);
    analogWrite(leftMotorPin2, 220);
    analogWrite(rightMotorPin1, 200);                                     
    analogWrite(rightMotorPin2, 0);
    //Serial.print("ok5");
  }  */
