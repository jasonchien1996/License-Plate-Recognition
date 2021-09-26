#include <string>
#include <array>
#include <unordered_map>
#include <iostream>

using namespace std;

class Car { 
private:
	int dominantCategory;
	array<unsigned, 3> category;
	array<unsigned, 3> confidence;    
	array<unsigned, 7> consec7;
	array<unsigned, 6> consec6;
	array<char, 7> currentSeven;
	array<char, 6> currentSix;
	array<unordered_map<char, unsigned>, 7> seven;
	array<unordered_map<char, unsigned>, 6> six;

	void increaseCategory(int category, int conf);
	bool loose_isalpha(string s);
	bool loose_isdigit(string s);
	bool strict_isalpha(string s);
	bool strict_isdigit(string s);
	char convert2alpha(char ch);
	char convert2digit(char ch);
	string strict2alpha(string s);
	string strict2digit(string s);    
	
public: 
	Car();
	array<unsigned, 7> getConsec7();
	array<unsigned, 6> getConsec6();
	string getPlate();
	string vote();
	void setCounter(string detection, int category, int conf, bool imply_category1);
	void setPlate(int index, char value);
	void resetConsecutive(int index);
	void increaseConsecutive(int index);
	bool recognize(string &result, int &category);	
	int getDominantCategory();	
};
