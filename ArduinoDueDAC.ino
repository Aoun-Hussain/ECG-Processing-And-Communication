
#include <DueTimer.h>
#include "wave.h"

#define ArrayLimit sizeof(waveTable)/sizeof(uint16_t)
//unsigned int period = 40;
int WaveNumber = 0;
void setup()
{
	Serial.begin(115200);
	analogWriteResolution(12);
	Timer3.attachInterrupt(SetDuty);
	Timer3.setFrequency(1000);
	Serial.println(ArrayLimit);

	Timer3.start();
	//analogWrite(DAC0, 4000);

}

// Add the main program code into the continuous loop() function

void loop()
{
	
	while (1)
	{

	}
}



void SetDuty(void)
{
	

		
		analogWrite(DAC0, (uint16_t)(pgm_read_word_near(waveTable + WaveNumber)));
		WaveNumber++;
		//pgm_read_dword_near
		//Serial.print((uint16_t)pgm_read_word_near(waveTable + WaveNumber));
		//Serial.print(", ");
		//Serial.println((uint16_t)pgm_read_word_near(waveTable + WaveNumber) * 2.7 / 4095.0);
		if (WaveNumber >= ArrayLimit)
		{
			
			WaveNumber = 0;
			//Timer1.detachInterrupt();

		}

}
