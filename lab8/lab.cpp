#include "mpi.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_set>
#include <ranges>
#include <algorithm>
#include <vector>

using namespace std;

// Разделяет строку str разделителем delim
vector<string> Split(const string &str, const string &delim)
{
    vector<string> split;
    for (const auto &s : views::split(str, delim))
    {
        split.emplace_back(s.begin(), s.end());
    }
    return split;
}

// Убирает по краям строки любые символы, определённые функцией сравнения. По умолчанию isspace (пробельные, в т.ч. \n)
string Trim(const string &in, int (*compare_func)(int) = isspace)
{
    auto sv = in |
              views::drop_while(compare_func) |
              views::reverse |
              views::drop_while(compare_func) |
              views::reverse;
    return string(sv.begin(), sv.end());
}

double duration(auto start_time, auto end_time)
{
    return std::chrono::duration<double>(end_time - start_time).count();
}

int main(int argc, char **argv)
{
    int rank;
    int total_processes;
    const int root = 0;

    string file_name(argc > 1 ? argv[1] : "text.txt");

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    string text;
    int text_len;

    if (rank == root)
    {
        auto file = ifstream(file_name);
        stringstream buf;
        buf << file.rdbuf();
        text = buf.str();
        transform(text.begin(), text.end(), text.begin(), [](char ch)
                  { return tolower(ch); });
        text_len = text.length();
    }
    
    string text_out;

    MPI_Bcast(&text_len, 1, MPI_INT, root, MPI_COMM_WORLD);

    int numbers_per_process = text_len / total_processes;
    if (text_len % total_processes != 0)
    {
        numbers_per_process++;
    }

    int displ[total_processes];
    int counts[total_processes];

    if (rank == root)
    {
        for (int i = 0; i < total_processes; ++i)
        {
            int s = numbers_per_process * i;
            displ[i] = s;
        }

        for (int i = 1; i < total_processes; ++i)
        {
            int shift = 0;
            int current_displ = displ[i];
            while (!isspace(text[current_displ + shift]) && (current_displ + shift < text_len))
            {
                ++shift;
            }
            displ[i] += shift + 1;
            counts[i - 1] = displ[i] - displ[i - 1];
        }
        counts[total_processes - 1] = text_len - displ[total_processes - 1];
    }

    MPI_Bcast(displ, total_processes, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(counts, total_processes, MPI_INT, root, MPI_COMM_WORLD);

    int len = counts[rank];
    string text_part;
    text_part.resize(len + 1);

    auto start_time = std::chrono::system_clock::now();
    MPI_Scatterv(text.data(), counts, displ, MPI_CHAR, text_part.data(), len, MPI_CHAR, root, MPI_COMM_WORLD);

    auto vec_split = Split(text_part, " ");
    
    auto trim_f = [](string str) {
        auto trim_char = [](int ch){
            return int(iscntrl(ch) || ispunct(ch));
        };
        return Trim(str, trim_char);
    };

    transform(vec_split.begin(), vec_split.end(), vec_split.begin(), trim_f);
    unordered_set<string> unique_part(vec_split.begin(), vec_split.end());

    stringstream u_sstr;
    for (auto& u: unique_part)
    {
        u_sstr << u << " ";
    }
    string u_str(u_sstr.str());
    len = u_str.length();

    MPI_Gather(&len, 1, MPI_INT, counts, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root)
    {
        int sum = 0;
        for (int i = 0; i < total_processes; ++i)
        {
            displ[i] = sum;
            sum += counts[i];
        }
        text_out.resize(sum+1);
    }

    MPI_Gatherv(u_str.data(), len, MPI_CHAR, text_out.data(), counts, displ, MPI_CHAR, root, MPI_COMM_WORLD);

    if (rank == root)
    {
        auto all_splt = Split(text_out, " ");
        transform(all_splt.begin(), all_splt.end(), all_splt.begin(), trim_f);
        unordered_set<string> unique(all_splt.begin(), all_splt.end());
        unique.erase("");
        cout << "Parallel: " << unique.size() << ", " << duration(start_time, std::chrono::system_clock::now()) << " sec." << endl;

        
        auto start_time = std::chrono::system_clock::now();
        auto all_splt2 = Split(text, " ");
        transform(all_splt2.begin(), all_splt2.end(), all_splt2.begin(), trim_f);
        unordered_set<string> unique2(all_splt2.begin(), all_splt2.end());
        unique2.erase("");
        cout << "Non-Parallel: " << unique2.size() << ", " << duration(start_time, std::chrono::system_clock::now()) << " sec." << endl;
    }

    MPI_Finalize();

    return 0;
}