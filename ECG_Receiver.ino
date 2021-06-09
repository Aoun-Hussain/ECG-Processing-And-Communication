#include <ESP8266WiFi.h> 
int a=0;    
void setup() {
  Serial.begin(9600);   //Serial.begin(115200);
  delay(10);
  WiFi.mode(WIFI_STA);
  WiFi.begin("New","12345678");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  //pinMode(7, INPUT); // Setup for leads off detection LO +
  //pinMode(8, INPUT); // Setup for leads off detection LO -
  }

void loop() {
  WiFiClient client;
  if (!client.connect("192.168.4.1",80)) {
    Serial.println("connection failed");
    return;
  }
  /*if((digitalRead(7) == 1)||(digitalRead(8) == 1)){
    Serial.println('!');
  }*/
  
  //String url = "GET /data/?button_state="+String(map(analogRead(A0),0,1023,0,255)); //analogRead(A0)*3.3/1024
  //Serial.println(map(analogRead(A0),0,1023,0,255));
  String url = "GET /data/?button_state="+String(analogRead(A0)*3.3/1023);
  Serial.println(analogRead(A0)*3.3/1023);
  client.print(url + " HTTP/1.1\r\nHost: 192.168.4.1\r\n" + "Connection: close\r\n\r\n");
  unsigned long timeout = millis();
  while (client.available() == 0) {
    if (millis() - timeout > 5000) {
      Serial.println(">>> Client Timeout !");
      client.stop();
      return;
    }}
    delay(2);
    }
