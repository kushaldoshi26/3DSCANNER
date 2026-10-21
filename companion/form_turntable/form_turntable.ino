// FORM controller, ULN2003 IN1..IN4 on D8..D11.
// The platter step count is calibrated in the companion, not assumed here.
const byte pins[4]={8,9,10,11};
const byte sequence[8][4]={{1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},{0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}};
long remaining=0;int phase=0,direction=1;unsigned long lastStep=0;
char command[40];byte used=0;
void releaseCoils(){for(byte i=0;i<4;i++)digitalWrite(pins[i],LOW);}
void processCommand(){
 command[used]=0;
 if(strcmp(command,"HELLO")==0)Serial.println("FORM_READY");
 else if(strcmp(command,"STOP")==0){remaining=0;releaseCoils();Serial.println("STOPPED");}
 else if(strncmp(command,"STEP ",5)==0){
   if(remaining){Serial.println("ERROR BUSY");return;}
   char* end;long count=strtol(command+5,&end,10);
   if(*end || count>500000L || count< -500000L){Serial.println("ERROR STEP_COUNT");return;}
   direction=count<0?-1:1;remaining=labs(count);
   if(!remaining)Serial.println("DONE");
 } else Serial.println("ERROR COMMAND");
}
void setup(){for(byte i=0;i<4;i++)pinMode(pins[i],OUTPUT);releaseCoils();Serial.begin(115200);}
void loop(){
 while(Serial.available()){
   char c=Serial.read();if(c=='\r')continue;
   if(c=='\n'){processCommand();used=0;}
   else if(used<sizeof(command)-1)command[used++]=c;
   else{used=0;Serial.println("ERROR TOO_LONG");}
 }
 if(remaining && micros()-lastStep>=2000UL){
   lastStep=micros();phase=(phase+direction+8)%8;
   for(byte i=0;i<4;i++)digitalWrite(pins[i],sequence[phase][i]);
   remaining--;if(!remaining){releaseCoils();Serial.println("DONE");}
 }
}
