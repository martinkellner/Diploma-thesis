#include <cstdio>
#include <yarp/os/SystemClock.h>

using namespace std;
using namespace yarp::os;

int main() {
    printf("Stating the application\n");
    int times=10;
    
    while (times--) {
        printf("Hello iCub\n");
        SystemClock::delaySystem(0.5);        
    }
    printf("Goodbey!\n");    
}
