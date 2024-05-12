---
layout: post
title:  "algorithms"
date:   2024-05-12 03:00:00 +0900
categories: c++ algorithm

---

---

# Index

[1. Dynamic programming](#1.-Dynamic-programming)

[2. Prefix sum](#2.-Prefix-sum)

[3. Greedy algorithm](#3.-Greedy-algorithm)

[4. Binary search](#4.-Binary-search)

[5. Divide and Conquer](#5.-Divide-and-Conquer)

[6. Priority queue](#6.-Priority-queue)

[7. Graph search](#7.-Graph-search)



---



# 1. Dynamic programming

- pseudo code (top-down)

```c++
int memo[100]{}; //메모이제이션 공간. 전역 변수이므로 0으로 초기화
int fibonacci(unsigned int n)
{
  if (n<=1) //0번째, 1번째 피보나치 수
    return n;
  if (memo[n]!=0) //메모가 있는지 확인(0으로 초기화되었으므로 0이 아니라면 메모가 쓰인 것임)
    return memo[n]; //메모 리턴
  memo[n]=fibonacci(n-1) + fibonacci(n-2); //작은 문제로 분할
  return memo[n];
}
```

- pseudo code (bottom-up)

```c++
int f_data[N] = {1, 1}; // N은 정의하기 나름
int last_pos = 1; // 마지막으로 계산한 지점. 이 코드에선 이미 f_data[1]까지 정의되어있기 때문에 1로 초기화한다.
int f(int n) //피보나치 수열의 제 n항을 구한다. 배열의 관점에서는 n-1번째 요소를 구하는 것.
{
    int i;
    if(f_data[n-1] == 0)  // 아직 구한 적이 없으면 구한다
    {
        for(i=last_pos+1; i<n; ++i)
        {
            f_data[i] = f_data[i-1] + f_data[i-2];
        }
        last_pos = n-1;
    }
    return f_data[n-1];
}
```





- outline

  최적화 이론의 한 기술

  앞서 계산한 결과를 재사용하여 불필요한 재귀 호출을 제거함.

  재사용을 위해 메모이제이션 활용함.

  최적 부분 구조 문제에서 매우 효과적인 알고리즘.

​    

- top down vs bottom up

  * bottom up

    일반적으로 bottom up이 선호된다.

    효율, 속도 측면에서 약간 우세하고 다차원 배열을 사용할 경우에도 어렵지 않게 해결가능하기 때문이다.

  * top down

    하지만 인간의 사고방식에 가까운 것은 top-down 방식이다. 자연스러운 접근이 가능함. bottom-up 방식은 코드짜는 것이 어려울수 있다.

​    

+ 결론

  bottom up을 기본으로 사용하고 아래의 경우 top down을 고려 :

  top down 방식을 사용하는 경우?

  bottom up 방식으로 접근했는데 서브문제의 해결이 어려운 경우. (점화식 모르겠을 때)

  서브문제 전체 크기에 비해 메인문제에 관여하는 수가 적을 때.

​    

* 문제 추천

  2579 계단오르기

  11053 가장 긴 증가하는 부분 수열 (LIS)

  2565 전깃줄

  9251 LCS

  12865 평범한 배낭 (knap sack)

​    

* 각 알고리즘 접근법

  LIS : 이중 for 문으로 0 < j < i < n 범위를 전부 탐색.

  LCS : 2개의 문자열의 공통부분 찾기, 2차원 배열과 특정 점화식을 이용해야함.

  Knap sack : n번째 품목을 포함하는 경우와 제외하는 경우에 대해 탐색.



# 2. Prefix sum



# 3. Greedy algorithm



# 4. Binary search



# 5. Divide and Conquer



# 6. Priority queue



# 7. Graph search



