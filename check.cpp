// // #include <iostream>
// // #include <map>
// // #include <set>
// // #include <sstream>
// // #include <vector>
// // #include <algorithm>
// // using namespace std;

// // struct Account {
// //     string userName;
// //     int balance;
// //     set<string> affiliatedBanks;

// //     // Constructors
// //     Account() : userName(""), balance(0) {}
// //     Account(const string& user, int bal, const vector<string>& banks) : userName(user), balance(bal) {
// //         affiliatedBanks.insert(banks.begin(), banks.end());
// //     }

// //     // Deposit money to account
// //     string deposit(int amount, const string& bank) {
// //         if (affiliatedBanks.count(bank) == 0)
// //             return "FAILURE";
// //         balance += amount;
// //         return "SUCCESS";
// //     }

// //     // Withdraw money from account
// //     string withdraw(int amount, const string& bank) {
// //         if (affiliatedBanks.count(bank) == 0 || balance < amount)
// //             return "FAILURE";
// //         balance -= amount;
// //         return "SUCCESS";
// //     }
// // };

// // // Splitting function to handle parsing
// // vector<string> tokenize(const string& input, char delimiter = ',') {
// //     vector<string> tokens;
// //     string token;
// //     istringstream tokenStream(input);
// //     while (getline(tokenStream, token, delimiter)) {
// //         tokens.push_back(token);
// //     }
// //     return tokens;
// // }

// // // Execute individual command logic
// // string handleCommand(const vector<string>& cmd, map<string, Account>& userAccounts, set<string>& globalBanks) {
// //     if (cmd[0] == "BALANCE") {
// //         if (userAccounts.count(cmd[2]) == 0)
// //             return "FAILURE";
// //         return to_string(userAccounts[cmd[2]].balance);
// //     } else if (cmd[0] == "TRANSFER") {
// //         string sender = cmd[2];
// //         string receiver = cmd[3];
// //         int amount = stoi(cmd[4]);

// //         if (globalBanks.count(sender) || globalBanks.count(receiver)) {
// //             return "FAILURE";
// //         }

// //         if (userAccounts.count(sender) == 0 || userAccounts.count(receiver) == 0) {
// //             return "FAILURE";
// //         }

// //         if (userAccounts[sender].balance < amount) {
// //             return "FAILURE";
// //         }

// //         userAccounts[sender].balance -= amount;
// //         userAccounts[receiver].balance += amount;
// //         return "SUCCESS";
// //     }
// //     return "INVALID";
// // }

// // // Process all commands
// // string processTransactions(const vector<string>& cmds) {
// //     map<string, Account> userAccounts;
// //     set<string> bankList;
// //     vector<pair<int, vector<string>>> sortedTransactions;
// //     vector<string> results;

// //     // Parse input commands
// //     for (const auto& cmdStr : cmds) {
// //         vector<string> command = tokenize(cmdStr);

// //         if (command[0] == "SETUP") {
// //             vector<string> banks(command.begin() + 3, command.end());
// //             userAccounts[command[1]] = Account(command[1], stoi(command[2]), banks);
// //             bankList.insert(banks.begin(), banks.end());
// //         } else {
// //             sortedTransactions.push_back({stoi(command[1]), command});
// //         }
// //     }

// //     // Sort based on time for processing
// //     sort(sortedTransactions.begin(), sortedTransactions.end());

// //     // Execute commands
// //     for (const auto& txn : sortedTransactions) {
// //         string result = handleCommand(txn.second, userAccounts, bankList);
// //         results.push_back(result);  // Store results in sequence
// //     }

// //     // Concatenate results
// //     stringstream finalResult;
// //     for (const auto& res : results) {
// //         if (!res.empty()) finalResult << res << ",\n";
// //     }
// //     string output = finalResult.str();
// //     if (!output.empty()) output.pop_back();
// //     return output;
// // }

// // int main() {
// //     vector<string> commands = {
// //         "SETUP,Alice,100,Chase,Wells Fargo",
// //         "SETUP,Bob,50,Bank of America,Chase,Ally",
// //         "SETUP,Charles,0,Bank of America",
// //         "TRANSFER,212,Alice,Bob,50",     // Alice sends 50 to Bob
// //         "BALANCE,211,Alice",             // Check Alice's balance before transaction
// //         "TRANSFER,213,Alice,Charles,50", // Alice attempts to send 50 to Charles (insufficient funds)
// //         "BALANCE,214,Alice",             // Alice's balance post-transaction
// //         "TRANSFER,301,Charles,Bob,100",  // Charles tries to send 100 but has insufficient balance
// //         "TRANSFER,302,Ally,Charles,10",  // Ally (a bank) tries to send 10 to Charles (invalid)
// //         "TRANSFER,305,Kate,Alice,50",    // Invalid user Kate attempts a transfer
// //         "BALANCE,401,Charles",           // Check Charles' balance (should be 0)
// //         "TRANSFER,306,Alice,Chase,50"    // Alice attempts to send 50 to Chase bank (invalid)
// //     };

// //     string output = processTransactions(commands);
// //     cout << output << endl;
// //     return 0;
// // }




// #include <iostream>
// #include <map>
// #include <set>
// #include <sstream>
// #include <vector>
// #include <algorithm>
// using namespace std;

// struct Account {
//     string userName;
//     int balance;
//     set<string> affiliatedBanks;

//     // Constructors
//     Account() : userName(""), balance(0) {}
//     Account(const string& user, int bal, const vector<string>& banks) : userName(user), balance(bal) {
//         affiliatedBanks.insert(banks.begin(), banks.end());
//     }

//     // Deposit money to account
//     string deposit(int amount, const string& bank) {
//         if (affiliatedBanks.count(bank) == 0)
//             return "FAILURE";
//         balance += amount;
//         return "SUCCESS";
//     }

//     // Withdraw money from account
//     string withdraw(int amount, const string& bank) {
//         if (affiliatedBanks.count(bank) == 0 || balance < amount)
//             return "FAILURE";
//         balance -= amount;
//         return "SUCCESS";
//     }
// };

// // Splitting function to handle parsing
// vector<string> tokenize(const string& input, char delimiter = ',') {
//     vector<string> tokens;
//     string token;
//     istringstream tokenStream(input);
//     while (getline(tokenStream, token, delimiter)) {
//         tokens.push_back(token);
//     }
//     return tokens;
// }

// // Execute individual command logic
// string handleCommand(const vector<string>& cmd, map<string, Account>& userAccounts, set<string>& globalBanks) {
//     if (cmd[0] == "BALANCE") {
//         if (userAccounts.count(cmd[2]) == 0)
//             return "FAILURE";
//         return to_string(userAccounts[cmd[2]].balance);
//     } else if (cmd[0] == "TRANSFER") {
//         string sender = cmd[2];
//         string receiver = cmd[3];
//         int amount = stoi(cmd[4]);

//         if (globalBanks.count(sender) || globalBanks.count(receiver)) {
//             return "FAILURE";
//         }

//         if (userAccounts.count(sender) == 0 || userAccounts.count(receiver) == 0) {
//             return "FAILURE";
//         }

//         if (userAccounts[sender].balance < amount) {
//             return "FAILURE";
//         }

//         userAccounts[sender].balance -= amount;
//         userAccounts[receiver].balance += amount;
//         return "SUCCESS";
//     }
//     return "INVALID";
// }

// // Process all commands
// string processTransactions(const vector<string>& cmds) {
//     map<string, Account> userAccounts;
//     set<string> bankList;
//     vector<pair<int, vector<string>>> sortedTransactions;
//     vector<string> results;

//     // Parse input commands
//     for (const auto& cmdStr : cmds) {
//         vector<string> command = tokenize(cmdStr);

//         if (command[0] == "SETUP") {
//             vector<string> banks(command.begin() + 3, command.end());
//             userAccounts[command[1]] = Account(command[1], stoi(command[2]), banks);
//             bankList.insert(banks.begin(), banks.end());
//         } else {
//             sortedTransactions.push_back({stoi(command[1]), command});
//         }
//     }

//     // Sort based on time for processing
//     sort(sortedTransactions.begin(), sortedTransactions.end());

//     // Execute commands
//     for (const auto& txn : sortedTransactions) {
//         string result = handleCommand(txn.second, userAccounts, bankList);
//         results.push_back(result);  // Store results in sequence
//     }

//     // Concatenate results
//     stringstream finalResult;
//     for (const auto& res : results) {
//         if (!res.empty()) finalResult << res << ",\n";
//     }
//     string output = finalResult.str();
//     if (!output.empty()) output.pop_back();
//     return output;
// }

// int main() {
//     vector<string> commands = {
//         "SETUP,Alice,100,Chase,Wells Fargo",
//         "SETUP,Bob,50,Bank of America,Chase,Ally",
//         "SETUP,Charles,0,Bank of America",
//         "TRANSFER,212,Alice,Bob,50",     // Alice sends 50 to Bob
//         "BALANCE,211,Alice",             // Check Alice's balance before transaction
//         "TRANSFER,213,Alice,Charles,50", // Alice attempts to send 50 to Charles (insufficient funds)
//         "BALANCE,214,Alice",             // Alice's balance post-transaction
//         "TRANSFER,301,Charles,Bob,100",  // Charles tries to send 100 but has insufficient balance
//         "TRANSFER,302,Ally,Charles,10",  // Ally (a bank) tries to send 10 to Charles (invalid)
//         "TRANSFER,305,Kate,Alice,50",    // Invalid user Kate attempts a transfer
//         "BALANCE,401,Charles",           // Check Charles' balance (should be 0)
//         "TRANSFER,306,Alice,Chase,50"    // Alice attempts to send 50 to Chase bank (invalid)
//     };

//     string output = processTransactions(commands);
//     cout << output << endl;
//     return 0;
// }

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

typedef pair<int, pair<int, int>> Cell;

int trapWater(vector<vector<int>>& heightMap, vector<pair<int, int>>& drains) {
    int m = heightMap.size();
    int n = heightMap[0].size();

    vector<vector<bool>> visited(m, vector<bool>(n, false));
    priority_queue<Cell, vector<Cell>, greater<Cell>> pq;

    // Add drain cells to the queue
    for (auto d : drains) {
        pq.push({heightMap[d.first][d.second], {d.first, d.second}});
        visited[d.first][d.second] = true;
    }

    // Add border cells
    for (int i = 0; i < m; i++) {
        if (!visited[i][0]) {
            pq.push({heightMap[i][0], {i, 0}});
            visited[i][0] = true;
        }
        if (!visited[i][n - 1]) {
            pq.push({heightMap[i][n - 1], {i, n - 1}});
            visited[i][n - 1] = true;
        }
    }

    for (int j = 0; j < n; j++) {
        if (!visited[0][j]) {
            pq.push({heightMap[0][j], {0, j}});
            visited[0][j] = true;
        }
        if (!visited[m - 1][j]) {
            pq.push({heightMap[m - 1][j], {m - 1, j}});
            visited[m - 1][j] = true;
        }
    }

    int water = 0;
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!pq.empty()) {
        Cell cell = pq.top();
        pq.pop();
        int height = cell.first;
        int x = cell.second.first;
        int y = cell.second.second;

        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d];
            int ny = y + dy[d];

            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                visited[nx][ny] = true;
                int neighborHeight = heightMap[nx][ny];
                if (neighborHeight < height) {
                    water += height - neighborHeight;
                    heightMap[nx][ny] = height; // Fill water to current height
                }
                pq.push({max(heightMap[nx][ny], height), {nx, ny}});
            }
        }
    }

    return water;
}

int main() {
    vector<vector<int>> elevation_map = {
        {1, 4, 3, 1, 3, 2},
        {3, 2, 1, 3, 2, 4},
        {2, 3, 3, 2, 3, 1}
    };

    vector<pair<int, int>> drains = {{2, 5}};

    cout << "Trapped Water: " << trapWater(elevation_map, drains) << endl;
    return 0;
}
