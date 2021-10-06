#include <string>
#include <array>
#include <unordered_map>
#include <iostream>

class Car { 
private:
	int dominantCategory;
	std::array<unsigned, 3> category;
	std::array<unsigned, 3> confidence;
	std::array<char, 7> currentSeven;
	std::array<char, 6> currentSix;
	std::array<std::unordered_map<char, unsigned>, 7> seven;
	std::array<std::unordered_map<char, unsigned>, 6> six;

	void increaseCategory(int category, int conf);
	bool loose_isalpha(std::string s);
	bool loose_isdigit(std::string s);
	bool strict_isalpha(std::string s);
	bool strict_isdigit(std::string s);
	char convert2alpha(char ch);
	char convert2digit(char ch);
	std::string strict2alpha(std::string s);
	std::string strict2digit(std::string s);

public: 
	Car();
	std::string vote();
	int getDominantCategory();	
	bool recognize(std::string &result, int &category);
	void setCounter(std::string detection, int category, int conf, bool imply_category1);
};
