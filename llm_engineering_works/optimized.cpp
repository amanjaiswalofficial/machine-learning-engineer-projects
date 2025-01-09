
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdint>

int main() {
  constexpr int iterations = 100;
  double param1 = 4.0;
  double param2 = 1.0;
  double result = 1.0;

  for (int i = 1; i <= iterations; ++i)
  {
    double j = i * param1 - param2;
    result -= (1.0 / j);
    j = i * param1 + param2;
    result += (1.0 / j);
  }
  // Calculate and print the result 

  std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();   
  double result1 = calculate(iterations, param1, param2) * 4;  
  std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::high_resolution_clock::now();  

  std::cout << "Result: " << std::setprecision(6) << result1 << std::endl;
  std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6 << " seconds" << std::endl;

}


double calculate(int iterations, double param1, double param2) {
  double result = 1.0;
    for (int i = 1; i <= iterations; ++i)
    {        
      double j = i * param1 - param2; 
      result -= (1 / j);   
      j = i * param1 + param2;
      result += (1 / j);
    }
    return result;
  }

