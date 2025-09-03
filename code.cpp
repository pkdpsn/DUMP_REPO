#include<iostream>
#include<vector>
using namespace std; 

int check(int nutsize, int boltsize){
    if (nutsize>boltsize){
        return -1;
    }
    else if (nutsize==boltsize){
        return 0;
    }
    return 1;
}

int binarysearch(vector<int>& piles , int nutsize , int index){
    int low = 0  , high = index , mid;
    while(low<= high){
        mid = (low+high)/2;
        int val = check(nutsize , piles[mid]);
        // cout<<"val "<<val<<endl;
        if (val==0){
            return mid;
            }
        else if (val ==-1){
            low =mid+1;
        }
        else high = mid-1;
        
    }
    return index+1;

}

int main (){
    vector<int> bolt = {1,2,3,4,5,6,7,8,9,10};
    vector<int> nut = {2,3,4,5,6,7,8,9,10,1};
    vector<int> pile(bolt.size(),0);
    vector<int>ans(nut.size(),0);
    int indexfilled=0;
    for(int i = 0 ; i< nut.size() ; i++ ){
        int newindex = binarysearch(bolt, nut[i] , indexfilled);
        // cout<<"index "<<newindex<<" nut"<<nut[i]<<" "<<i <<endl;
        // cout<<nut[]
        // newindex =< indexfile 
        if (newindex<= nut.size()){
            ans[newindex] = nut[i];
            }  
        if (indexfilled<newindex){
            indexfilled = newindex;
        }
        
    }
    for (int i = 0 ; i<nut.size() ; i++){
        cout<<ans[i]<<" ";
    } 
    return 0;


}