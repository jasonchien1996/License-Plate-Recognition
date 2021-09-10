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

Car::Car() {
	dominantCategory = -1;
	category = {0, 0, 0};
	confidence = {0, 0, 0};
	currentSeven = {'#', '#', '#', '#', '#', '#', '#'};
	currentSix = {'#', '#', '#', '#', '#', '#'};
	consec7 = {0, 0, 0, 0, 0, 0, 0};
	consec6 = {0, 0, 0, 0, 0, 0};
}

bool Car::recognize(string &s, int &category) {
	/*
	category 0:7-character
    category 1:6-character(2+4)
    category 2:6-character(4+2)
    category 3:6-character(unkown)
	*/
	bool imply_category1 = false;
	if ( s.length() == 8 ) {
		//cout << "l == 8" << endl;
		/*
		little possibility that the length of the plate is less than 7
    	convert the string to a 7-character string only when the dominant category is already a 7-character one
		*/
    	if ( this->dominantCategory == 0 ) {
			string left = strict2alpha(s.substr(0,3)) + strict2digit(s.substr(3,4));
            string right = strict2alpha(s.substr(1,3)) + strict2digit(s.substr(4,4));
			unsigned short l = 0;
			unsigned short r = 0;
			for ( int i = 0; i < 7; ++i ) {
                if ( left[i] == this->currentSeven[i] )
                    l += 1;
                if ( right[i] == this->currentSeven[i] )
                    r += 1;
			}
			if ( l >= r && l > 3 )
                s = left;
            else if ( l < r and r > 3 )
                s = right;
			else
				s = "";
        }
		else
			s = "";
	}

	else if ( s.length() == 7 ) {
        //cout << "l == 7" << endl;
        category = 0;
        s = strict2alpha(s.substr(0,3)) + strict2digit(s.substr(3,4));
	}
	
	else if ( s.length() == 6 ) {
        //cout << "l == 6" << endl;
		
        //check conditions from strict ones to loose ones
        if ( (this->strict_isalpha(s.substr(0,1)) || this->strict_isalpha(s.substr(1,1))) && this->strict_isdigit(s.substr(4,2)) ) {
          	category = 1;
			//cout << "1" << endl;		
		}
        else if ( (this->strict_isalpha(s.substr(4,1)) || this->strict_isalpha(s.substr(5,1))) && this->strict_isdigit(s.substr(0,2)) ) {                
			category = 2;
			//cout << "2" << endl;
		}
		else if ( this->strict_isalpha(s.substr(0,2)) ){
			category = 1;		
		} 
		else if ( this->strict_isalpha(s.substr(4,2)) ){
			category = 2;		
		}
        //loose condition in number part
		else if ( (this->strict_isalpha(s.substr(0,1)) || this->strict_isalpha(s.substr(1,1))) && this->loose_isdigit(s.substr(4,2)) ) {
            category = 1;
			//cout << "3" << endl;	
		}
        else if ( (this->strict_isalpha(s.substr(4,1)) || this->strict_isalpha(s.substr(5,1))) && this->loose_isdigit(s.substr(0,2)) ) {
            category = 2;
			//cout << "4" << endl;	                      
		}
        //loose condition in letter part
        else if ( (this->loose_isalpha(s.substr(0,1)) || this->loose_isalpha(s.substr(1,1))) && this->strict_isdigit(s.substr(4,2)) ) {
            category = 1;
			//cout << "5" << endl;
		}	
        else if ( (this->loose_isalpha(s.substr(4,1)) || this->loose_isalpha(s.substr(5,1))) && this->strict_isdigit(s.substr(0,2)) ) {
            category = 2;          
        	//cout << "6" << endl;
		}	
        else
            category = 3;

        if ( category == 1 ) {
			if ( s[0] == '0' ) s[0] = 'O';
			if ( s[1] == '0' ) s[1] = 'O';
			if ( s[0] == '1' ) s[0] = 'I';
			if ( s[1] == '1' ) s[1] = 'I';
            s = s.substr(0,2) + strict2digit(s.substr(2,4));
			imply_category1 = true;
        }
        else if ( category == 2 ) {
			if ( s[4] == '0' ) s[4] = 'O';
			if ( s[5] == '0' ) s[5] = 'O';
			if ( s[4] == '1' ) s[4] = 'I';
			if ( s[5] == '1' ) s[5] = 'I';
            s = strict2digit(s.substr(0,4)) + s.substr(4,2);			
		}		
        else
            s = s.substr(0,2) + strict2digit(s.substr(2,2)) + s.substr(4,2);
	}

	else if ( s.length() == 5 ) {
		//cout << "l == 5" << endl;
		//dominant category is 0
        if ( this->dominantCategory == 0 ) {
            string left = this->strict2alpha(s.substr(0,3)) + strict2digit(s.substr(3,2)) + "##";
            string middle = "#" + strict2alpha(s.substr(0,2)) + strict2digit(s.substr(2,3)) + "#";
            string right = "##" + strict2alpha(s.substr(0,1)) + strict2digit(s.substr(1,4));
            unsigned short l = 0;
            unsigned short m = 0;
            unsigned short r = 0;
            for ( int i = 0; i < 7; ++i ) {
                if ( left[i] == this->currentSeven[i] )
                    l += 1;
                if ( middle[i] == this->currentSeven[i] )
                    m += 1;
                if ( right[i] == this->currentSeven[i] )
                    r += 1;
			}
            if ( l > m && l > r && l > 3 )
                s = left;
            else if ( m > l && m > r && m > 3 )
                s = middle;
            else if ( r > l && r > m && r > 3 )
                s = right;
			else
				s = "";
		}
        //dominant category is 1
        else if ( this->category[1] > 2*this->category[2] ) {
            // adddd
            if ( this->strict_isalpha(s.substr(0,1)) && this->strict_isdigit(s.substr(1,1)) && this->loose_isdigit(s.substr(2,3)) )
                s = "#" + s[0] + this->strict2digit(s.substr(1,4));
            // ?addd
            else if ( this->strict_isalpha(s.substr(1,1)) && this->strict_isdigit(s.substr(2,1)) && this->loose_isdigit(s.substr(3,2)) )
                s = s.substr(0,3)+ this->strict2digit(s.substr(3,2)) + "#";
			else
				s = "";
			imply_category1 = true;
        }
        //dominant category is 2
        else if ( 2*this->category[1] < this->category[2] ) {
            // dddda
            if ( this->loose_isdigit(s.substr(0,3)) && this->strict_isdigit(s.substr(3,1)) && this->strict_isalpha(s.substr(4,1)) )
                s = strict2digit(s.substr(0,4)) + s[4] + "#";
            // ddda?
            else if ( this->loose_isdigit(s.substr(0,2)) && this->strict_isdigit(s.substr(2,1)) && this->strict_isalpha(s.substr(3,1)) )
                s = "#" + strict2digit(s.substr(0,3)) + s.substr(3,2);
			else
				s = "";
		}
		else
			s = "";
	}

	else if ( s.length() == 4 ) {
        //cout << "l == 4" << endl;
        //dominant category is 0
        if ( this->dominantCategory == 0 ) {
            // dddd
            if ( this->strict_isdigit(s.substr(0,1)) && this->loose_isdigit(s.substr(1,3)) )
                s = "###" + this->strict2digit(s.substr(0,4));
            // ad??
            else if ( this->strict_isalpha(s.substr(0,1)) && this->strict_isdigit(s.substr(1,1)) )
                s = "##" + s[0] + this->strict2digit(s.substr(1,3)) + "#";
            // ?ad?
            else if ( this->strict_isalpha(s.substr(1,1)) && this->strict_isdigit(s.substr(2,1)) )
                s = "#" + this->strict2alpha(s.substr(0,2)) + this->strict2digit(s.substr(2,2)) + "##";
            // ??ad
            else if ( this->strict_isalpha(s.substr(2,1)) && this->strict_isdigit(s.substr(3,1)) )
                s = this->strict2alpha(s.substr(0,3)) + s[3] + "###";
			else
				s = "";
        }
        //dominant category is 1
        else if ( this->category[1] > 2*this->category[2] ) {
            // dddd
            if ( this->strict_isdigit(s.substr(0,1)) && this->loose_isdigit(s.substr(1,3)) )
                s = "##" + this->strict2digit(s.substr(0,4));
            // ?add
            else if ( this->strict_isalpha(s.substr(1,1)) && this->strict_isdigit(s.substr(2,1)) && this->loose_isdigit(s.substr(3,1)) )
                s = s.substr(0,2) + this->strict2digit(s.substr(0,2)) + "##";
			else
				s = "";
			imply_category1 = true;
		}
        //dominant category is 2
        else if ( 2*this->category[1] < this->category[2] ) {
            // dddd
            if ( this->strict_isdigit(s.substr(0,1)) && this->loose_isdigit(s.substr(1,3)) )
                s = this->strict2digit(s.substr(0,4)) + "##";
            // dda?
            else if ( this->loose_isdigit(s.substr(0,1)) && this->strict_isdigit(s.substr(1,1)) && this->strict_isalpha(s.substr(2,1)) )
                s = "##" + strict2digit(s.substr(0,2)) + s.substr(2,2);
			else
				s = "";
		}
	}

	else {
		cout << "error in recognize()\n";
		exit(1);	
	}

	return imply_category1;
}

void Car::setCounter(string detection, int category, int conf, bool imply_category1) {
	char ch;    
	if ( detection.length() == 6 ) { 
        for ( int i = 0; i < 6; ++i ) {
            ch = detection[i];                              
            if ( ch != '#' ) {
				if ( this->six[i].find(ch) == this->six[i].end() )
                	this->six[i][ch] = 1;
				else
					this->six[i][ch] += 1;
                if ( i > 1 && imply_category1 ) { //only for car-plate
					if ( this->seven[i].find(ch) == this->seven[i].end() )
                		this->seven[i+1][ch] = 1;
					else
                    	this->seven[i+1][ch] += 1;
				}
			}
		}
	}
    else if ( detection.length() == 7 ) {
        for ( int i = 0; i < 7; ++i ) {
            ch = detection[i];   
            if ( ch != '#' ) {
                if ( this->seven[i].find(ch) == this->seven[i].end() )
                	this->seven[i][ch] = 1;
				else
                	this->seven[i][ch] += 1;
			}
		}
	}
    if ( category == 0 || category == 1 || category == 2 )
        this->increaseCategory(category, conf);
    else if ( category == 3 ) {
        this->increaseCategory(1, conf);
        this->increaseCategory(2, conf);
	}
}

void Car::increaseCategory(int category, int conf) {
	this->category[category] += 1;
	this->confidence[category] += conf;
	if ( this->dominantCategory == -1 ) {
		if ( this->category[0] > this->category[1] && this->category[0] > this->category[2] ) this->dominantCategory = 0;
		else if ( this->category[1] > this->category[0] && this->category[1] > this->category[2] ) this->dominantCategory = 1;
		else this->dominantCategory = 2;		
	}
	else{
		unsigned short argMax = 0;
		if ( this->category[1] > this->category[argMax] ) argMax = 1;
		if ( this->category[2] > this->category[argMax] ) argMax = 2;
		if ( this->category[argMax] > this->category[this->dominantCategory] ) this->dominantCategory = argMax;        
        else if ( this->category[argMax] == this->category[this->dominantCategory] ) {
            if ( this->confidence[argMax] > this->confidence[this->dominantCategory] ) this->dominantCategory = argMax;
		}
	}	
}

int Car::getDominantCategory(){
    return this->dominantCategory;
}

string Car::getPlate() {
    if ( this->dominantCategory == 0) { 
		string plate(std::begin(this->currentSeven), std::end(this->currentSeven));
        return plate;
	}
    else if ( this->dominantCategory == 1 || this->dominantCategory == 2 ) {
		string plate(std::begin(this->currentSix), std::end(this->currentSix));
        return plate;
	}
    else {
		cout <<"error in getPlate()\n";
		exit(1);
	}
}

string Car::vote() {
	string res = "";
    if ( this->dominantCategory == 0 ) {
        for ( int i = 0; i < 7; ++i ) {
			unsigned currentMax = 0;
			char arg_max = '#';
			for ( auto iter = seven[i].cbegin(); iter != seven[i].cend(); ++iter ) {
				if ( iter->second > currentMax ) {
					arg_max = iter->first;
					currentMax = iter->second;
				}
			}
			res += arg_max;
		}
	}
    else if ( this->dominantCategory == 1 || this->dominantCategory == 2 ) {
        for ( int i = 0; i < 6; ++i ) {
			unsigned currentMax = 0;
			char arg_max = '#';
			for ( auto iter = six[i].cbegin(); iter != six[i].cend(); ++iter ) {
				if ( iter->second > currentMax ) {
					arg_max = iter->first;
					currentMax = iter->second;
				}
			}
			res += arg_max;
		}
	}
    return res;
}

array<unsigned, 7> Car::getConsec7() {
	return this->consec7;
}

array<unsigned, 6> Car::getConsec6() {
	return this->consec6;
}

void Car::setPlate(int index, char value) {
    if ( this->dominantCategory == 0 )
        this->currentSeven[index] = value;
    else if ( this->dominantCategory == 1 || this->dominantCategory == 2 ) 
        this->currentSix[index] = value;
    else {
		cout << "error in setPlate()\n";
		exit(1);
	}
}

void Car::resetConsecutive(int index) {
    if ( this->dominantCategory == 0 )
        this->consec7[index] = 1;
    else if ( this->dominantCategory == 1 || this->dominantCategory == 2)
        this->consec6[index] = 1;
    else {
        cout << "error in resetConsecutive()\n";
		exit(1);
	}
}

void Car::increaseConsecutive(int index){
    if ( this->dominantCategory == 0 )
        this->consec7[index] += 1;
    else if ( this->dominantCategory == 1 || this->dominantCategory == 2 )
        this->consec6[index] += 1;
    else {
		cout << "error in increaseConsecutive()\n";
		exit(1);
	}
}

bool Car::loose_isalpha(string s) {
	for(char ch : s){
		if(isalpha(ch) == 0 && ch != '0' && ch != '1')
        	return false;
	}
    return true;
}

bool Car::loose_isdigit(string s) {
	for(char ch : s){
        if(isdigit(ch) == 0 && ch != 'O' && ch != 'I' && ch != 'D')
	        return false;
	}
    return true;
}

bool Car::strict_isalpha(string s) {
	for(char ch : s){
		if(isalpha(ch) == 0 || ch == 'O' || ch == 'I')
        	return false;
	}
    return true;
}

bool Car::strict_isdigit(string s) {
	for(char ch : s){
        if(isdigit(ch) == 0 || ch == '0' || ch == '1')
	        return false;
	}
    return true;
}

char Car::convert2alpha(char ch) {
    if( ch == '0')
        return 'O';
    if( ch == '1')
        return 'I';
    if( ch == '2')
        return 'Z';
    if( ch == '3')
        return 'B';
    if( ch == '4')
        return 'A';
    if( ch == '5')
        return 'S';
    if( ch == '6')
        return 'G';
    if( ch == '7')
        return 'J';
    if( ch == '8')
        return 'B';
    return '#';
}       

char Car::convert2digit(char ch) {
    if( ch == 'A')
        return '4';
    if( ch == 'B')
        return '8';
    if( ch == 'C')
        return '6';
    if( ch == 'D')
        return '0';
    if( ch == 'G')
        return '6';
    if( ch == 'I')
        return '1';
    if( ch == 'J')
        return '7';
    if( ch == 'L')
        return '2';
    if( ch == 'O')
        return '0';
    if( ch == 'Q')
        return '0';
    if( ch == 'R')
        return '8';
    if( ch == 'S')
        return '5';
    if( ch == 'T')
        return '7';
    if( ch == 'Z')
        return '2';
    return '#';
}

string Car::strict2alpha(string s) {
	string res = "";    
	for(char ch : s){
        if(isalpha(ch) == 0)
            ch = this->convert2alpha(ch);
		res += ch;
	}
	return res;
}

string Car::strict2digit(string s) {
	string res = ""; 
    for(char ch : s){
        if(isdigit(ch) == 0)
            ch = this->convert2digit(ch);				
		res += ch;
	}
	return res;
}

