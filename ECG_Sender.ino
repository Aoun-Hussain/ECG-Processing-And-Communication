#include <ESP8266WiFi.h> 
int a=0;    
void setup() {
  Serial.begin(115200);
  delay(10);
  WiFi.mode(WIFI_STA);
  WiFi.begin("New","12345678");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  }

void loop() {
  WiFiClient client;
  if (!client.connect("192.168.4.1",80)) {
    Serial.println("connection failed");
    return;
  }
  String url = "GET /data/?button_state="+String(analogRead(A0)*3.3/1024);
  Serial.println(analogRead(A0)*3.3/1024);
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
