---
layout: post
title:  "algorithms"
date:   2024-05-12 03:00:00 +0900
categories: c++ algorithm
---

a summary of algorithms

# Index

[1. Dynamic programming](#1.-Dynamic-programming)

[2. Prefix sum](#2.-Prefix-sum)

[3. Greedy algorithm](#3.-Greedy-algorithm)

[4. Binary search](#4.-Binary-search)

[5. Divide and Conquer](#5.-Divide-and-Conquer)

[6. Priority queue](#6.-Priority-queue)

[7. Graph search](#7.-Graph-search)

[8. Dijkstra's algorithm]

[9. Floyd warshall algorithm]



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
  
  

- top down vs bottom up

  * bottom up

    일반적으로 bottom up이 선호된다.

    효율, 속도 측면에서 약간 우세하고 다차원 배열을 사용할 경우에도 어렵지 않게 해결가능하기 때문이다.

  * top down

    하지만 인간의 사고방식에 가까운 것은 top-down 방식이다. 자연스러운 접근이 가능함. bottom-up 방식은 코드짜는 것이 어려울수 있다.
    
    

+ 결론

  bottom up을 기본으로 사용하고 아래의 경우 top down을 고려 :

  top down 방식을 사용하는 경우?

  bottom up 방식으로 접근했는데 서브문제의 해결이 어려운 경우. (점화식 모르겠을 때)

  서브문제 전체 크기에 비해 메인문제에 관여하는 수가 적을 때.
  
  

* problems

  2579 계단오르기

  11053 가장 긴 증가하는 부분 수열 (LIS)

  2565 전깃줄

  9251 LCS

  12865 평범한 배낭 (knap sack)
  
  

* 문제별 접근법

  LIS : 이중 for 문으로 0 < j < i < n 범위를 전부 탐색.

  LCS : 2개의 문자열의 공통부분 찾기, 2차원 배열과 특정 점화식을 이용해야함.

  Knap sack : n번째 품목을 포함하는 경우와 제외하는 경우에 대해 탐색.



---



# 2. Prefix sum

- pseudo code

```c++
prefix()
{
	for(i=0; i<=n; i++)
		prefix_sum[i] = prefix_sum[i-1] + number[i];	
}

subsum(int i, int j)
{
    return prefix_sum[j] - prefix_sum[i];
}
```

- outline

  특정 구간의 합을 구해야 할 때 고려할 수 있는 알고리즘.

  처음에 모든 수에 대해 누적합을 구해두고 누적합의 차를 이용하여 구간의 합을 구하는 방법.

  

- algorithm

  1. `sum(a,b)` 는 a에서 b까지 합
  2. 모든 i 에 대해 `sum(0,i)` 를 구한다.
  3. 구간 `i ~ j` 의 합 = `sum(0,j)` - `sum(0,i-1)`



- problems

  11659 구간 합 구하기 4 (기본개념)

  16139 인간-컴퓨터 상호작용 (2차원 배열로 확장)

  10986 나머지 합 (실수 다발)



---



# 3. Greedy algorithm

- outline

  현재 상태에서 알 수 있는 정보만을 이용하여 최선의 선택을 하는 알고리즘

  다른 상황을 고려하지 않고 현 시점에서 최선의 선택을 하는 기법

  이 알고리즘이 최선인 경우는 다음 두 조건이 만족될 때 이다.

  1. greedy choice property : 앞의 선택이 이후의 선택에 영향을 주지 않음
  2. optimal substructure : 문제의 최적해가 부분 문제에서도 최적해인 경우

---



# 4. Binary search

- pseudo code

```c++
// initialize
left = 최소값
right = 최대값

while(left <= right)
{
	// 중간값 선택
	mid = (left + right) / 2; 

	// 값 비교후, 범위 재조정.
	if(mid == value)
		break;
	else if (mid < value)
		left = mid + 1;
	else
		right = mid - 1;
}

return mid;
```

- outline

  **정렬된 값**에서 특정 값을 찾기 위해 사용하는 알고리즘.

  중간 지점의 값과 찾을 값을 비교하고 다음 범위를 선택함을 반복하여 탐색범위를 절반씩 줄여감.

  O(logn)

  

- 주의할 점

  반드시 정렬된 값에서만 가능

  범위 지정 주의하기. 무한루프 발생 가능

---



# 5. Divide and Conquer

- outline

  문제를 해결 가능한 수준의 작은 문제들로 분할하고, 각 문제를 해결한 값을 합쳐 최종 해를 구하는 알고리즘.

  보통 재귀함수로 구현한다.

  divide -> conquer -> combine

  

- advantages :

  병렬적 해결에서 강점

  어려운 문제를 해결하기 위해 고려할 수 있는 알고리즘

  

- disadvantages :

  재귀함수의 문제점을 가짐

  문제를 최소 단위로 분할하는 기준이 중요함.

  퍼포먼스에 상당한 영향을 주는데, 적절한 기준을 선정하는 것이 어려운 문제점이 있음.



- problems

  6549 히스토그램에서 가장 큰 직사각형



---



# 6. Priority queue

- header \<queue>

```c++
#include <iostream>
#include <algorithm>
#include <queue>

struct cmp
{
    bool operator() (pair<int,int> &a, pair<int,int> &b)
    {
        return a.first < b.first;
    }
}

int main()
{
    // 선언
    priority_queue<int> pq1; // 선언 (default : 오름차순)
    priority_queue<int, vector<int>, less<int>> pq2; // 정수 내림차순으로 선언
    priority_queue<int, vector<int>, greater<int>> pq3; // 정수 오름차순으로 선언
    
    // 연산자 오버로딩이용, 원하는 정렬기준을 만들 수 있음
    priority_queue<pair<int,int>, vector<pair<int,int>>, cmp> pq4;
        
    // methods
    pq1.push(1);
    pq1.pop();
    pq1.top();
    pq1.empty();
    pq1.size();
    
}
```



- outline

  우선순위 큐는 값들이 오름차순 또는 내림차순으로 push, pop 되는 큐이다.

  push, pop은 O(logn) 에 실행된다.

  보통 구현시 배열을 이용한다.



---



# 7. Graph search

## 7.1. DFS

- c++ code

```c++
void dfs(const vector<int> graph[], int current)
{ 
    visited[current] = true;

    for(int next: graph[current]) 
    { 
        if(!visited[next]) 
            dfs(graph, next); 
    }
}
```



- outline

  DFS : depth-first search

  스택 또는 재귀함수로 구현

  경로의 특징을 저장해야 하는 경우 사용함



## 7.2. BFS

- c++ code

```c++
// adjacent 인접 리스트, start 시작 노드
void bfs(const vector<int> adjacent[], int start) 
{ 
    // start 방문
    q.push(start);
    visited[start] = true;

    // 큐가 빌 때까지 반복
    while(!q.empty()) 
    { 
        // 큐에서 노드 하나를 꺼냄
        int current = q.front();
        q.pop();

        // current의 인접 노드 : next
        for(int next: adjacent[current]) 
        { 
            // 만일 next에 방문하지 않았다면 방문
            if(!visited[next]) 
            { 
                q.push(next);
                visited[next] = true;
            }
        }
    }
}
```



- outline

  BFS : breadth-first search

  큐로 구현

  최단 거리를 구할 때 사용함



---



# 8. Dijkstra's algorithm

- c++ code

```c++
// 1..N 노드가 임의의 가중치가 부여된 방향이 없는 간선으로 연결되어있음
// 다음 코드는 start 노드에서 출발하여, 모든 노드에 대한 최단 거리를 구하는 코드임
void dijkstra(int start)
{
  	// 거리 초기화, start에서 모든 노드까지 거리를 INF로 초기화함.
	init();
	
    // priority queue
	// pair<target, distance> 를 인자로 함
	// 현 시점 start ~ target 사이 최단 경로인 distance를 기록
	// distance 가 최소인 값이 top에 오는 형태로, 거리가 가까운 노드부터 탐색하고자함
	priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> priority_que;

	priority_que.push({ start,0 });
	distance_start_to[start] = 0;

	while (!priority_que.empty())
	{
		int current = priority_que.top().first;
		int current_dist = priority_que.top().second;
		priority_que.pop();

		// 탐색할 필요없는 케이스
		if (distance_start_to[current] < current_dist)
			continue;

		// current 에 이웃한 노드
		for (int i = 0; i < adjacent[current].size(); i++)
		{
			int next = adjacent[current][i].first;
			int next_dist = adjacent[current][i].second;

			// 거리갱신
			if (distance_start_to[next] > distance_start_to[current] + next_dist)
			{
				distance_start_to[next] = distance_start_to[current] + next_dist;
				priority_que.push({ next,distance_start_to[current] + next_dist });
			}
		}
	}
}
```



- outline

  다익스트라 알고리즘은 한 노드로 부터 나머지 노드까지 최단거리를 구하는 알고리즘이다. (one-to-all)

  구현시 priority queue 를 사용한다.

  *O*(*ElogE*+*ElogE*) 의 시간복잡도를 가짐



- algorithm

  0. 노드 A 와 B 사이 거리를 `P(A,B)`라고 가정한다. 

  1. 출발점에서 다른 노드까지 거리를 저장할 배열을 만들고 INF 로 초기화. 출발점에서 A 까지 거리 = `d(A)`
  2. 현재 노드를 저장하는 변수 `current` 에 출발할 노드를 할당한다.
  3. `current`에 이웃한 노드 `adjacent_node` 에 대해, `d(current) + P(current,adjacent_node)` < `d(adjacent_node)` 인 경우 : `d(adjacent_node)` 를 다음 값으로 갱신한다 :`d(current) + P(current,adjacent_node)`.
  4. `current`에 이웃한 모든 노드 `adjacent_node`에 대해 `3.`을 반복한다.
  5. `current`를 `방문완료` 상태로 바꾼다.
  6. 미방문노드  `K` 중 `d(K)`가 최소인 노드를 찾고 `current`에 할당한다.
  7. 도착노드가 방문완료 상태가 되거나, 더 이상의 미방문 노드가 없을때까지 `3.` ~ `6.` 을 반복한다.

---



# 9. Floyd-Warshall algorithm

- c++ code

```c++
void Floyd()
{
	for (int m = 1; m <= n; m++) //가운데 노드
		for (int s = 1; s <= n; s++) //시작 노드
			for (int e = 1; e <= n; e++) //마지막 노드
				if (d[s][e] > d[s][m] + d[m][e])
					d[s][e] = d[s][m] + d[m][e]; //가운데를 거쳐가는 것이 더 빠르면 그걸로 업데이트한다.
}
```

- outline

  모든 노드에서 모든 노드로의 최단거리를 구하는 알고리즘

  O(V^3)

  s 에서 e 로 가는 최단 거리를 구하기 위해 중간노드 m 을 지나는 모든 경우를 비교

  `d(s,e) = min(d(s,e), d(s,m) + d(m,e))`

- algorithm

  1. 노드 a 부터 노드 b 까지의 최단 거리를 저장할 배열을 만듦. d\[V][V] (방향그래프)

  2. 배열 d 를 초기화.

     2.1. `if(i==j) : d[i][j] = 0`

     2.2. `if(i!=j) : d[i][j] = INF`

  3. 모든 m에 대해 m을 지나가는 모든 경우를 확인함. m이 3중 for문의 가장 위에 있어야함. 

     `d(s,e) = min(d(s,e), d(s,m) + d(m,e))`

  
