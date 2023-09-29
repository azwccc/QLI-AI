/*
The software is provided "as is," without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement.
In no event shall the authors or copyright holders be liable for any claim,
damages or other liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the software."
*/

#ifndef AVX512
#define AVX512 0
#endif // AVX512

#define PORT 21841
#define EPOCH 71

#include <cassert>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#include <stdio.h>
#include <string.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <cstring>
#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <chrono>
#include <thread>
#endif

#if not(defined(_WIN32) || defined(_WIN64))
[[maybe_unused]] static __inline__ char _InterlockedCompareExchange8(char volatile *_Destination, char _Exchange, char _Comparand)
{
    __atomic_compare_exchange(_Destination, &_Comparand, &_Exchange, 0, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
    return _Comparand;
}
#endif

#define ACQUIRE(lock)                                 \
    while (_InterlockedCompareExchange8(&lock, 1, 0)) \
    _mm_pause()
#define RELEASE(lock) lock = 0
#define EQUAL(a, b) ((unsigned)(_mm256_movemask_epi8(_mm256_cmpeq_epi64(a, b))) == 0xFFFFFFFF)

#if defined(_MSC_VER)
#define ROL64(a, offset) _rotl64(a, offset)
#else
#define ROL64(a, offset) ((((unsigned long long)a) << offset) ^ (((unsigned long long)a) >> (64 - offset)))
#endif

// for intel based mac. RELEASE_X86_64 x86_64 . tested on xnu-7195
#if defined(__APPLE__) && defined(__MACH__)
void explicit_bzero(void *b, size_t len)
{
    memset_s(b, len, 0, len);
}
#endif

#if AVX512
const static __m512i zero = _mm512_maskz_set1_epi64(0, 0);
const static __m512i moveThetaPrev = _mm512_setr_epi64(4, 0, 1, 2, 3, 5, 6, 7);
const static __m512i moveThetaNext = _mm512_setr_epi64(1, 2, 3, 4, 0, 5, 6, 7);
const static __m512i rhoB = _mm512_setr_epi64(0, 1, 62, 28, 27, 0, 0, 0);
const static __m512i rhoG = _mm512_setr_epi64(36, 44, 6, 55, 20, 0, 0, 0);
const static __m512i rhoK = _mm512_setr_epi64(3, 10, 43, 25, 39, 0, 0, 0);
const static __m512i rhoM = _mm512_setr_epi64(41, 45, 15, 21, 8, 0, 0, 0);
const static __m512i rhoS = _mm512_setr_epi64(18, 2, 61, 56, 14, 0, 0, 0);
const static __m512i pi1B = _mm512_setr_epi64(0, 3, 1, 4, 2, 5, 6, 7);
const static __m512i pi1G = _mm512_setr_epi64(1, 4, 2, 0, 3, 5, 6, 7);
const static __m512i pi1K = _mm512_setr_epi64(2, 0, 3, 1, 4, 5, 6, 7);
const static __m512i pi1M = _mm512_setr_epi64(3, 1, 4, 2, 0, 5, 6, 7);
const static __m512i pi1S = _mm512_setr_epi64(4, 2, 0, 3, 1, 5, 6, 7);
const static __m512i pi2S1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 8, 10);
const static __m512i pi2S2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 9, 11);
const static __m512i pi2BG = _mm512_setr_epi64(0, 1, 8, 9, 6, 5, 6, 7);
const static __m512i pi2KM = _mm512_setr_epi64(2, 3, 10, 11, 7, 5, 6, 7);
const static __m512i pi2S3 = _mm512_setr_epi64(4, 5, 12, 13, 4, 5, 6, 7);
const static __m512i padding = _mm512_maskz_set1_epi64(1, 0x8000000000000000);

const static __m512i K12RoundConst0 = _mm512_maskz_set1_epi64(1, 0x000000008000808bULL);
const static __m512i K12RoundConst1 = _mm512_maskz_set1_epi64(1, 0x800000000000008bULL);
const static __m512i K12RoundConst2 = _mm512_maskz_set1_epi64(1, 0x8000000000008089ULL);
const static __m512i K12RoundConst3 = _mm512_maskz_set1_epi64(1, 0x8000000000008003ULL);
const static __m512i K12RoundConst4 = _mm512_maskz_set1_epi64(1, 0x8000000000008002ULL);
const static __m512i K12RoundConst5 = _mm512_maskz_set1_epi64(1, 0x8000000000000080ULL);
const static __m512i K12RoundConst6 = _mm512_maskz_set1_epi64(1, 0x000000000000800aULL);
const static __m512i K12RoundConst7 = _mm512_maskz_set1_epi64(1, 0x800000008000000aULL);
const static __m512i K12RoundConst8 = _mm512_maskz_set1_epi64(1, 0x8000000080008081ULL);
const static __m512i K12RoundConst9 = _mm512_maskz_set1_epi64(1, 0x8000000000008080ULL);
const static __m512i K12RoundConst10 = _mm512_maskz_set1_epi64(1, 0x0000000080000001ULL);
const static __m512i K12RoundConst11 = _mm512_maskz_set1_epi64(1, 0x8000000080008008ULL);

#else

#define KeccakF1600RoundConstant0 0x000000008000808bULL
#define KeccakF1600RoundConstant1 0x800000000000008bULL
#define KeccakF1600RoundConstant2 0x8000000000008089ULL
#define KeccakF1600RoundConstant3 0x8000000000008003ULL
#define KeccakF1600RoundConstant4 0x8000000000008002ULL
#define KeccakF1600RoundConstant5 0x8000000000000080ULL
#define KeccakF1600RoundConstant6 0x000000000000800aULL
#define KeccakF1600RoundConstant7 0x800000008000000aULL
#define KeccakF1600RoundConstant8 0x8000000080008081ULL
#define KeccakF1600RoundConstant9 0x8000000000008080ULL
#define KeccakF1600RoundConstant10 0x0000000080000001ULL

#define declareABCDE                            \
    unsigned long long Aba, Abe, Abi, Abo, Abu; \
    unsigned long long Aga, Age, Agi, Ago, Agu; \
    unsigned long long Aka, Ake, Aki, Ako, Aku; \
    unsigned long long Ama, Ame, Ami, Amo, Amu; \
    unsigned long long Asa, Ase, Asi, Aso, Asu; \
    unsigned long long Bba, Bbe, Bbi, Bbo, Bbu; \
    unsigned long long Bga, Bge, Bgi, Bgo, Bgu; \
    unsigned long long Bka, Bke, Bki, Bko, Bku; \
    unsigned long long Bma, Bme, Bmi, Bmo, Bmu; \
    unsigned long long Bsa, Bse, Bsi, Bso, Bsu; \
    unsigned long long Ca, Ce, Ci, Co, Cu;      \
    unsigned long long Da, De, Di, Do, Du;      \
    unsigned long long Eba, Ebe, Ebi, Ebo, Ebu; \
    unsigned long long Ega, Ege, Egi, Ego, Egu; \
    unsigned long long Eka, Eke, Eki, Eko, Eku; \
    unsigned long long Ema, Eme, Emi, Emo, Emu; \
    unsigned long long Esa, Ese, Esi, Eso, Esu;

#define thetaRhoPiChiIotaPrepareTheta(i, A, E) \
    Da = Cu ^ ROL64(Ce, 1);                    \
    De = Ca ^ ROL64(Ci, 1);                    \
    Di = Ce ^ ROL64(Co, 1);                    \
    Do = Ci ^ ROL64(Cu, 1);                    \
    Du = Co ^ ROL64(Ca, 1);                    \
    A##ba ^= Da;                               \
    Bba = A##ba;                               \
    A##ge ^= De;                               \
    Bbe = ROL64(A##ge, 44);                    \
    A##ki ^= Di;                               \
    Bbi = ROL64(A##ki, 43);                    \
    A##mo ^= Do;                               \
    Bbo = ROL64(A##mo, 21);                    \
    A##su ^= Du;                               \
    Bbu = ROL64(A##su, 14);                    \
    E##ba = Bba ^ ((~Bbe) & Bbi);              \
    E##ba ^= KeccakF1600RoundConstant##i;      \
    Ca = E##ba;                                \
    E##be = Bbe ^ ((~Bbi) & Bbo);              \
    Ce = E##be;                                \
    E##bi = Bbi ^ ((~Bbo) & Bbu);              \
    Ci = E##bi;                                \
    E##bo = Bbo ^ ((~Bbu) & Bba);              \
    Co = E##bo;                                \
    E##bu = Bbu ^ ((~Bba) & Bbe);              \
    Cu = E##bu;                                \
    A##bo ^= Do;                               \
    Bga = ROL64(A##bo, 28);                    \
    A##gu ^= Du;                               \
    Bge = ROL64(A##gu, 20);                    \
    A##ka ^= Da;                               \
    Bgi = ROL64(A##ka, 3);                     \
    A##me ^= De;                               \
    Bgo = ROL64(A##me, 45);                    \
    A##si ^= Di;                               \
    Bgu = ROL64(A##si, 61);                    \
    E##ga = Bga ^ ((~Bge) & Bgi);              \
    Ca ^= E##ga;                               \
    E##ge = Bge ^ ((~Bgi) & Bgo);              \
    Ce ^= E##ge;                               \
    E##gi = Bgi ^ ((~Bgo) & Bgu);              \
    Ci ^= E##gi;                               \
    E##go = Bgo ^ ((~Bgu) & Bga);              \
    Co ^= E##go;                               \
    E##gu = Bgu ^ ((~Bga) & Bge);              \
    Cu ^= E##gu;                               \
    A##be ^= De;                               \
    Bka = ROL64(A##be, 1);                     \
    A##gi ^= Di;                               \
    Bke = ROL64(A##gi, 6);                     \
    A##ko ^= Do;                               \
    Bki = ROL64(A##ko, 25);                    \
    A##mu ^= Du;                               \
    Bko = ROL64(A##mu, 8);                     \
    A##sa ^= Da;                               \
    Bku = ROL64(A##sa, 18);                    \
    E##ka = Bka ^ ((~Bke) & Bki);              \
    Ca ^= E##ka;                               \
    E##ke = Bke ^ ((~Bki) & Bko);              \
    Ce ^= E##ke;                               \
    E##ki = Bki ^ ((~Bko) & Bku);              \
    Ci ^= E##ki;                               \
    E##ko = Bko ^ ((~Bku) & Bka);              \
    Co ^= E##ko;                               \
    E##ku = Bku ^ ((~Bka) & Bke);              \
    Cu ^= E##ku;                               \
    A##bu ^= Du;                               \
    Bma = ROL64(A##bu, 27);                    \
    A##ga ^= Da;                               \
    Bme = ROL64(A##ga, 36);                    \
    A##ke ^= De;                               \
    Bmi = ROL64(A##ke, 10);                    \
    A##mi ^= Di;                               \
    Bmo = ROL64(A##mi, 15);                    \
    A##so ^= Do;                               \
    Bmu = ROL64(A##so, 56);                    \
    E##ma = Bma ^ ((~Bme) & Bmi);              \
    Ca ^= E##ma;                               \
    E##me = Bme ^ ((~Bmi) & Bmo);              \
    Ce ^= E##me;                               \
    E##mi = Bmi ^ ((~Bmo) & Bmu);              \
    Ci ^= E##mi;                               \
    E##mo = Bmo ^ ((~Bmu) & Bma);              \
    Co ^= E##mo;                               \
    E##mu = Bmu ^ ((~Bma) & Bme);              \
    Cu ^= E##mu;                               \
    A##bi ^= Di;                               \
    Bsa = ROL64(A##bi, 62);                    \
    A##go ^= Do;                               \
    Bse = ROL64(A##go, 55);                    \
    A##ku ^= Du;                               \
    Bsi = ROL64(A##ku, 39);                    \
    A##ma ^= Da;                               \
    Bso = ROL64(A##ma, 41);                    \
    A##se ^= De;                               \
    Bsu = ROL64(A##se, 2);                     \
    E##sa = Bsa ^ ((~Bse) & Bsi);              \
    Ca ^= E##sa;                               \
    E##se = Bse ^ ((~Bsi) & Bso);              \
    Ce ^= E##se;                               \
    E##si = Bsi ^ ((~Bso) & Bsu);              \
    Ci ^= E##si;                               \
    E##so = Bso ^ ((~Bsu) & Bsa);              \
    Co ^= E##so;                               \
    E##su = Bsu ^ ((~Bsa) & Bse);              \
    Cu ^= E##su;

#define copyFromState(state) \
    Aba = state[0];          \
    Abe = state[1];          \
    Abi = state[2];          \
    Abo = state[3];          \
    Abu = state[4];          \
    Aga = state[5];          \
    Age = state[6];          \
    Agi = state[7];          \
    Ago = state[8];          \
    Agu = state[9];          \
    Aka = state[10];         \
    Ake = state[11];         \
    Aki = state[12];         \
    Ako = state[13];         \
    Aku = state[14];         \
    Ama = state[15];         \
    Ame = state[16];         \
    Ami = state[17];         \
    Amo = state[18];         \
    Amu = state[19];         \
    Asa = state[20];         \
    Ase = state[21];         \
    Asi = state[22];         \
    Aso = state[23];         \
    Asu = state[24];

#define copyToState(state) \
    state[0] = Aba;        \
    state[1] = Abe;        \
    state[2] = Abi;        \
    state[3] = Abo;        \
    state[4] = Abu;        \
    state[5] = Aga;        \
    state[6] = Age;        \
    state[7] = Agi;        \
    state[8] = Ago;        \
    state[9] = Agu;        \
    state[10] = Aka;       \
    state[11] = Ake;       \
    state[12] = Aki;       \
    state[13] = Ako;       \
    state[14] = Aku;       \
    state[15] = Ama;       \
    state[16] = Ame;       \
    state[17] = Ami;       \
    state[18] = Amo;       \
    state[19] = Amu;       \
    state[20] = Asa;       \
    state[21] = Ase;       \
    state[22] = Asi;       \
    state[23] = Aso;       \
    state[24] = Asu;

#define rounds12                                                                    \
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;                                               \
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;                                               \
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;                                               \
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;                                               \
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;                                               \
    thetaRhoPiChiIotaPrepareTheta(0, A, E)                                          \
        thetaRhoPiChiIotaPrepareTheta(1, E, A)                                      \
            thetaRhoPiChiIotaPrepareTheta(2, A, E)                                  \
                thetaRhoPiChiIotaPrepareTheta(3, E, A)                              \
                    thetaRhoPiChiIotaPrepareTheta(4, A, E)                          \
                        thetaRhoPiChiIotaPrepareTheta(5, E, A)                      \
                            thetaRhoPiChiIotaPrepareTheta(6, A, E)                  \
                                thetaRhoPiChiIotaPrepareTheta(7, E, A)              \
                                    thetaRhoPiChiIotaPrepareTheta(8, A, E)          \
                                        thetaRhoPiChiIotaPrepareTheta(9, E, A)      \
                                            thetaRhoPiChiIotaPrepareTheta(10, A, E) \
                                                Da = Cu ^ ROL64(Ce, 1);             \
    De = Ca ^ ROL64(Ci, 1);                                                         \
    Di = Ce ^ ROL64(Co, 1);                                                         \
    Do = Ci ^ ROL64(Cu, 1);                                                         \
    Du = Co ^ ROL64(Ca, 1);                                                         \
    Eba ^= Da;                                                                      \
    Bba = Eba;                                                                      \
    Ege ^= De;                                                                      \
    Bbe = ROL64(Ege, 44);                                                           \
    Eki ^= Di;                                                                      \
    Bbi = ROL64(Eki, 43);                                                           \
    Emo ^= Do;                                                                      \
    Bbo = ROL64(Emo, 21);                                                           \
    Esu ^= Du;                                                                      \
    Bbu = ROL64(Esu, 14);                                                           \
    Aba = Bba ^ ((~Bbe) & Bbi);                                                     \
    Aba ^= 0x8000000080008008ULL;                                                   \
    Abe = Bbe ^ ((~Bbi) & Bbo);                                                     \
    Abi = Bbi ^ ((~Bbo) & Bbu);                                                     \
    Abo = Bbo ^ ((~Bbu) & Bba);                                                     \
    Abu = Bbu ^ ((~Bba) & Bbe);                                                     \
    Ebo ^= Do;                                                                      \
    Bga = ROL64(Ebo, 28);                                                           \
    Egu ^= Du;                                                                      \
    Bge = ROL64(Egu, 20);                                                           \
    Eka ^= Da;                                                                      \
    Bgi = ROL64(Eka, 3);                                                            \
    Eme ^= De;                                                                      \
    Bgo = ROL64(Eme, 45);                                                           \
    Esi ^= Di;                                                                      \
    Bgu = ROL64(Esi, 61);                                                           \
    Aga = Bga ^ ((~Bge) & Bgi);                                                     \
    Age = Bge ^ ((~Bgi) & Bgo);                                                     \
    Agi = Bgi ^ ((~Bgo) & Bgu);                                                     \
    Ago = Bgo ^ ((~Bgu) & Bga);                                                     \
    Agu = Bgu ^ ((~Bga) & Bge);                                                     \
    Ebe ^= De;                                                                      \
    Bka = ROL64(Ebe, 1);                                                            \
    Egi ^= Di;                                                                      \
    Bke = ROL64(Egi, 6);                                                            \
    Eko ^= Do;                                                                      \
    Bki = ROL64(Eko, 25);                                                           \
    Emu ^= Du;                                                                      \
    Bko = ROL64(Emu, 8);                                                            \
    Esa ^= Da;                                                                      \
    Bku = ROL64(Esa, 18);                                                           \
    Aka = Bka ^ ((~Bke) & Bki);                                                     \
    Ake = Bke ^ ((~Bki) & Bko);                                                     \
    Aki = Bki ^ ((~Bko) & Bku);                                                     \
    Ako = Bko ^ ((~Bku) & Bka);                                                     \
    Aku = Bku ^ ((~Bka) & Bke);                                                     \
    Ebu ^= Du;                                                                      \
    Bma = ROL64(Ebu, 27);                                                           \
    Ega ^= Da;                                                                      \
    Bme = ROL64(Ega, 36);                                                           \
    Eke ^= De;                                                                      \
    Bmi = ROL64(Eke, 10);                                                           \
    Emi ^= Di;                                                                      \
    Bmo = ROL64(Emi, 15);                                                           \
    Eso ^= Do;                                                                      \
    Bmu = ROL64(Eso, 56);                                                           \
    Ama = Bma ^ ((~Bme) & Bmi);                                                     \
    Ame = Bme ^ ((~Bmi) & Bmo);                                                     \
    Ami = Bmi ^ ((~Bmo) & Bmu);                                                     \
    Amo = Bmo ^ ((~Bmu) & Bma);                                                     \
    Amu = Bmu ^ ((~Bma) & Bme);                                                     \
    Ebi ^= Di;                                                                      \
    Bsa = ROL64(Ebi, 62);                                                           \
    Ego ^= Do;                                                                      \
    Bse = ROL64(Ego, 55);                                                           \
    Eku ^= Du;                                                                      \
    Bsi = ROL64(Eku, 39);                                                           \
    Ema ^= Da;                                                                      \
    Bso = ROL64(Ema, 41);                                                           \
    Ese ^= De;                                                                      \
    Bsu = ROL64(Ese, 2);                                                            \
    Asa = Bsa ^ ((~Bse) & Bsi);                                                     \
    Ase = Bse ^ ((~Bsi) & Bso);                                                     \
    Asi = Bsi ^ ((~Bso) & Bsu);                                                     \
    Aso = Bso ^ ((~Bsu) & Bsa);                                                     \
    Asu = Bsu ^ ((~Bsa) & Bse);
#endif

#define K12_security 128
#define K12_capacity (2 * K12_security)
#define K12_capacityInBytes (K12_capacity / 8)
#define K12_rateInBytes ((1600 - K12_capacity) / 8)
#define K12_chunkSize 8192
#define K12_suffixLeaf 0x0B

typedef struct
{
    unsigned char state[200];
    unsigned char byteIOIndex;
} KangarooTwelve_F;

static void KeccakP1600_Permute_12rounds(unsigned char *state)
{
#if AVX512
    __m512i Baeiou = _mm512_maskz_loadu_epi64(0x1F, state);
    __m512i Gaeiou = _mm512_maskz_loadu_epi64(0x1F, state + 40);
    __m512i Kaeiou = _mm512_maskz_loadu_epi64(0x1F, state + 80);
    __m512i Maeiou = _mm512_maskz_loadu_epi64(0x1F, state + 120);
    __m512i Saeiou = _mm512_maskz_loadu_epi64(0x1F, state + 160);
    __m512i b0, b1, b2, b3, b4, b5;

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst0);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst1);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst2);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst3);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst4);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst5);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst6);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst7);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst8);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst9);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst10);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst11);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    _mm512_mask_storeu_epi64(state, 0x1F, Baeiou);
    _mm512_mask_storeu_epi64(state + 40, 0x1F, Gaeiou);
    _mm512_mask_storeu_epi64(state + 80, 0x1F, Kaeiou);
    _mm512_mask_storeu_epi64(state + 120, 0x1F, Maeiou);
    _mm512_mask_storeu_epi64(state + 160, 0x1F, Saeiou);
#else
    declareABCDE unsigned long long *stateAsLanes = (unsigned long long *)state;
    copyFromState(stateAsLanes)
        rounds12
        copyToState(stateAsLanes)
#endif
}

static void KangarooTwelve_F_Absorb(KangarooTwelve_F *instance, unsigned char *data, unsigned long long dataByteLen)
{
    unsigned long long i = 0;
    while (i < dataByteLen)
    {
        if (!instance->byteIOIndex && dataByteLen >= i + K12_rateInBytes)
        {
#if AVX512
            __m512i Baeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state);
            __m512i Gaeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 40);
            __m512i Kaeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 80);
            __m512i Maeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 120);
            __m512i Saeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 160);
#else
            declareABCDE unsigned long long *stateAsLanes = (unsigned long long *)instance->state;
            copyFromState(stateAsLanes)
#endif
            unsigned long long modifiedDataByteLen = dataByteLen - i;
            while (modifiedDataByteLen >= K12_rateInBytes)
            {
#if AVX512
                Baeiou = _mm512_xor_si512(Baeiou, _mm512_maskz_loadu_epi64(0x1F, data));
                Gaeiou = _mm512_xor_si512(Gaeiou, _mm512_maskz_loadu_epi64(0x1F, data + 40));
                Kaeiou = _mm512_xor_si512(Kaeiou, _mm512_maskz_loadu_epi64(0x1F, data + 80));
                Maeiou = _mm512_xor_si512(Maeiou, _mm512_maskz_loadu_epi64(0x1F, data + 120));
                Saeiou = _mm512_xor_si512(Saeiou, _mm512_maskz_loadu_epi64(0x01, data + 160));
                __m512i b0, b1, b2, b3, b4, b5;

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst0);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst1);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst2);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst3);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst4);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst5);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst6);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst7);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst8);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst9);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst10);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst11);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);
#else
                Aba ^= ((unsigned long long *)data)[0];
                Abe ^= ((unsigned long long *)data)[1];
                Abi ^= ((unsigned long long *)data)[2];
                Abo ^= ((unsigned long long *)data)[3];
                Abu ^= ((unsigned long long *)data)[4];
                Aga ^= ((unsigned long long *)data)[5];
                Age ^= ((unsigned long long *)data)[6];
                Agi ^= ((unsigned long long *)data)[7];
                Ago ^= ((unsigned long long *)data)[8];
                Agu ^= ((unsigned long long *)data)[9];
                Aka ^= ((unsigned long long *)data)[10];
                Ake ^= ((unsigned long long *)data)[11];
                Aki ^= ((unsigned long long *)data)[12];
                Ako ^= ((unsigned long long *)data)[13];
                Aku ^= ((unsigned long long *)data)[14];
                Ama ^= ((unsigned long long *)data)[15];
                Ame ^= ((unsigned long long *)data)[16];
                Ami ^= ((unsigned long long *)data)[17];
                Amo ^= ((unsigned long long *)data)[18];
                Amu ^= ((unsigned long long *)data)[19];
                Asa ^= ((unsigned long long *)data)[20];
                rounds12
#endif
                data += K12_rateInBytes;
                modifiedDataByteLen -= K12_rateInBytes;
            }
#if AVX512
            _mm512_mask_storeu_epi64(instance->state, 0x1F, Baeiou);
            _mm512_mask_storeu_epi64(instance->state + 40, 0x1F, Gaeiou);
            _mm512_mask_storeu_epi64(instance->state + 80, 0x1F, Kaeiou);
            _mm512_mask_storeu_epi64(instance->state + 120, 0x1F, Maeiou);
            _mm512_mask_storeu_epi64(instance->state + 160, 0x1F, Saeiou);
#else
            copyToState(stateAsLanes)
#endif
            i = dataByteLen - modifiedDataByteLen;
        }
        else
        {
            unsigned char partialBlock;
            if ((dataByteLen - i) + instance->byteIOIndex > K12_rateInBytes)
            {
                partialBlock = K12_rateInBytes - instance->byteIOIndex;
            }
            else
            {
                partialBlock = (unsigned char)(dataByteLen - i);
            }
            i += partialBlock;

            if (!instance->byteIOIndex)
            {
                unsigned int j = 0;
                for (; (j + 8) <= (unsigned int)(partialBlock >> 3); j += 8)
                {
                    ((unsigned long long *)instance->state)[j + 0] ^= ((unsigned long long *)data)[j + 0];
                    ((unsigned long long *)instance->state)[j + 1] ^= ((unsigned long long *)data)[j + 1];
                    ((unsigned long long *)instance->state)[j + 2] ^= ((unsigned long long *)data)[j + 2];
                    ((unsigned long long *)instance->state)[j + 3] ^= ((unsigned long long *)data)[j + 3];
                    ((unsigned long long *)instance->state)[j + 4] ^= ((unsigned long long *)data)[j + 4];
                    ((unsigned long long *)instance->state)[j + 5] ^= ((unsigned long long *)data)[j + 5];
                    ((unsigned long long *)instance->state)[j + 6] ^= ((unsigned long long *)data)[j + 6];
                    ((unsigned long long *)instance->state)[j + 7] ^= ((unsigned long long *)data)[j + 7];
                }
                for (; (j + 4) <= (unsigned int)(partialBlock >> 3); j += 4)
                {
                    ((unsigned long long *)instance->state)[j + 0] ^= ((unsigned long long *)data)[j + 0];
                    ((unsigned long long *)instance->state)[j + 1] ^= ((unsigned long long *)data)[j + 1];
                    ((unsigned long long *)instance->state)[j + 2] ^= ((unsigned long long *)data)[j + 2];
                    ((unsigned long long *)instance->state)[j + 3] ^= ((unsigned long long *)data)[j + 3];
                }
                for (; (j + 2) <= (unsigned int)(partialBlock >> 3); j += 2)
                {
                    ((unsigned long long *)instance->state)[j + 0] ^= ((unsigned long long *)data)[j + 0];
                    ((unsigned long long *)instance->state)[j + 1] ^= ((unsigned long long *)data)[j + 1];
                }
                if (j < (unsigned int)(partialBlock >> 3))
                {
                    ((unsigned long long *)instance->state)[j + 0] ^= ((unsigned long long *)data)[j + 0];
                }
                if (partialBlock & 7)
                {
                    unsigned long long lane = 0;
                    memcpy(&lane, data + (partialBlock & 0xFFFFFFF8), partialBlock & 7);
                    ((unsigned long long *)instance->state)[partialBlock >> 3] ^= lane;
                }
            }
            else
            {
                unsigned int _sizeLeft = partialBlock;
                unsigned int _lanePosition = instance->byteIOIndex >> 3;
                unsigned int _offsetInLane = instance->byteIOIndex & 7;
                const unsigned char *_curData = data;
                while (_sizeLeft > 0)
                {
                    unsigned int _bytesInLane = 8 - _offsetInLane;
                    if (_bytesInLane > _sizeLeft)
                    {
                        _bytesInLane = _sizeLeft;
                    }
                    if (_bytesInLane)
                    {
                        unsigned long long lane = 0;
                        memcpy(&lane, (void *)_curData, _bytesInLane);
                        ((unsigned long long *)instance->state)[_lanePosition] ^= (lane << (_offsetInLane << 3));
                    }
                    _sizeLeft -= _bytesInLane;
                    _lanePosition++;
                    _offsetInLane = 0;
                    _curData += _bytesInLane;
                }
            }

            data += partialBlock;
            instance->byteIOIndex += partialBlock;
            if (instance->byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(instance->state);
                instance->byteIOIndex = 0;
            }
        }
    }
}

static void KangarooTwelve(unsigned char *input, unsigned int inputByteLen, unsigned char *output, unsigned int outputByteLen)
{
    KangarooTwelve_F queueNode;
    KangarooTwelve_F finalNode;
    unsigned int blockNumber, queueAbsorbedLen;

    memset(&finalNode, 0, sizeof(KangarooTwelve_F));
    const unsigned int len = inputByteLen ^ ((K12_chunkSize ^ inputByteLen) & -(K12_chunkSize < inputByteLen));
    KangarooTwelve_F_Absorb(&finalNode, input, len);
    input += len;
    inputByteLen -= len;
    if (len == K12_chunkSize && inputByteLen)
    {
        blockNumber = 1;
        queueAbsorbedLen = 0;
        finalNode.state[finalNode.byteIOIndex] ^= 0x03;
        if (++finalNode.byteIOIndex == K12_rateInBytes)
        {
            KeccakP1600_Permute_12rounds(finalNode.state);
            finalNode.byteIOIndex = 0;
        }
        else
        {
            finalNode.byteIOIndex = (finalNode.byteIOIndex + 7) & ~7;
        }

        while (inputByteLen > 0)
        {
            const unsigned int len = K12_chunkSize ^ ((inputByteLen ^ K12_chunkSize) & -(inputByteLen < K12_chunkSize));
            memset(&queueNode, 0, sizeof(KangarooTwelve_F));
            KangarooTwelve_F_Absorb(&queueNode, input, len);
            input += len;
            inputByteLen -= len;
            if (len == K12_chunkSize)
            {
                ++blockNumber;
                queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
                queueNode.state[K12_rateInBytes - 1] ^= 0x80;
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = K12_capacityInBytes;
                KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
            }
            else
            {
                queueAbsorbedLen = len;
            }
        }

        if (queueAbsorbedLen)
        {
            if (++queueNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = 0;
            }
            if (++queueAbsorbedLen == K12_chunkSize)
            {
                ++blockNumber;
                queueAbsorbedLen = 0;
                queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
                queueNode.state[K12_rateInBytes - 1] ^= 0x80;
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = K12_capacityInBytes;
                KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
            }
        }
        else
        {
            memset(queueNode.state, 0, sizeof(queueNode.state));
            queueNode.byteIOIndex = 1;
            queueAbsorbedLen = 1;
        }
    }
    else
    {
        if (len == K12_chunkSize)
        {
            blockNumber = 1;
            finalNode.state[finalNode.byteIOIndex] ^= 0x03;
            if (++finalNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(finalNode.state);
                finalNode.byteIOIndex = 0;
            }
            else
            {
                finalNode.byteIOIndex = (finalNode.byteIOIndex + 7) & ~7;
            }

            memset(queueNode.state, 0, sizeof(queueNode.state));
            queueNode.byteIOIndex = 1;
            queueAbsorbedLen = 1;
        }
        else
        {
            blockNumber = 0;
            if (++finalNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(finalNode.state);
                finalNode.state[0] ^= 0x07;
            }
            else
            {
                finalNode.state[finalNode.byteIOIndex] ^= 0x07;
            }
        }
    }

    if (blockNumber)
    {
        if (queueAbsorbedLen)
        {
            blockNumber++;
            queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
            queueNode.state[K12_rateInBytes - 1] ^= 0x80;
            KeccakP1600_Permute_12rounds(queueNode.state);
            KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
        }
        unsigned int n = 0;
        for (unsigned long long v = --blockNumber; v && (n < sizeof(unsigned long long)); ++n, v >>= 8)
        {
        }
        unsigned char encbuf[sizeof(unsigned long long) + 1 + 2];
        for (unsigned int i = 1; i <= n; ++i)
        {
            encbuf[i - 1] = (unsigned char)(blockNumber >> (8 * (n - i)));
        }
        encbuf[n] = (unsigned char)n;
        encbuf[++n] = 0xFF;
        encbuf[++n] = 0xFF;
        KangarooTwelve_F_Absorb(&finalNode, encbuf, ++n);
        finalNode.state[finalNode.byteIOIndex] ^= 0x06;
    }
    finalNode.state[K12_rateInBytes - 1] ^= 0x80;
    KeccakP1600_Permute_12rounds(finalNode.state);
    memcpy(output, finalNode.state, outputByteLen);
}

static void KangarooTwelve64To32(unsigned char* input, unsigned char* output)
{
#if AVX512
    __m512i Baeiou = _mm512_maskz_loadu_epi64(0x1F, input);
    __m512i Gaeiou = _mm512_set_epi64(0, 0, 0, 0, 0x0700, ((unsigned long long*)input)[7], ((unsigned long long*)input)[6], ((unsigned long long*)input)[5]);

    __m512i b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, zero, 0x96), zero, padding, 0x96);
    __m512i b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    __m512i b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(zero, b0, b1, 0x96), rhoK));
    __m512i b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(zero, b0, b1, 0x96), rhoM));
    __m512i b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(padding, b0, b1, 0x96), rhoS));
    __m512i b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst0);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    __m512i Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    __m512i Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    __m512i Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst1);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst2);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst3);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst4);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst5);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst6);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst7);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst8);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst9);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst10);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));

    _mm512_mask_storeu_epi64(output, 0xF, _mm512_permutex2var_epi64(_mm512_permutex2var_epi64(_mm512_unpacklo_epi64(_mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst11), _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2)), pi2S1, _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2)), pi2BG, _mm512_unpacklo_epi64(_mm512_ternarylogic_epi64(b2, b3, b4, 0xD2), _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2))));
#else
    unsigned long long Aba, Abe, Abi, Abo, Abu;
    unsigned long long Aga, Age, Agi, Ago, Agu;
    unsigned long long Aka, Ake, Aki, Ako, Aku;
    unsigned long long Ama, Ame, Ami, Amo, Amu;
    unsigned long long Asa, Ase, Asi, Aso, Asu;
    unsigned long long Bba, Bbe, Bbi, Bbo, Bbu;
    unsigned long long Bga, Bge, Bgi, Bgo, Bgu;
    unsigned long long Bka, Bke, Bki, Bko, Bku;
    unsigned long long Bma, Bme, Bmi, Bmo, Bmu;
    unsigned long long Bsa, Bse, Bsi, Bso, Bsu;
    unsigned long long Ca, Ce, Ci, Co, Cu;
    unsigned long long Da, De, Di, Do, Du;
    unsigned long long Eba, Ebe, Ebi, Ebo, Ebu;
    unsigned long long Ega, Ege, Egi, Ego, Egu;
    unsigned long long Eka, Eke, Eki, Eko, Eku;
    unsigned long long Ema, Eme, Emi, Emo, Emu;
    unsigned long long Esa, Ese, Esi, Eso, Esu;

    Ca = ((unsigned long long*)input)[0] ^ ((unsigned long long*)input)[5] ^ 0x8000000000000000;
    Ce = ((unsigned long long*)input)[1] ^ ((unsigned long long*)input)[6];
    Ci = ((unsigned long long*)input)[2] ^ ((unsigned long long*)input)[7];
    Co = ((unsigned long long*)input)[3] ^ 0x0700;

    Da = ((unsigned long long*)input)[4] ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(((unsigned long long*)input)[4], 1);
    Du = Co ^ ROL64(Ca, 1);
    Aba = ((unsigned long long*)input)[0] ^ Da;
    Bbe = ROL64(((unsigned long long*)input)[6] ^ De, 44);
    Bbi = ROL64(Di, 43);
    Bbo = ROL64(Do, 21);
    Bbu = ROL64(Du, 14);
    Eba = Aba ^ __andn_u64(Bbe, Bbi) ^ 0x000000008000808bULL;
    Ebe = Bbe ^ __andn_u64(Bbi, Bbo);
    Ebi = Bbi ^ __andn_u64(Bbo, Bbu);
    Ebo = Bbo ^ __andn_u64(Bbu, Aba);
    Ebu = Bbu ^ __andn_u64(Aba, Bbe);
    Bga = ROL64(((unsigned long long*)input)[3] ^ Do, 28);
    Bge = ROL64(Du, 20);
    Bgi = ROL64(Da, 3);
    Bgo = ROL64(De, 45);
    Bgu = ROL64(Di, 61);
    Ega = Bga ^ __andn_u64(Bge, Bgi);
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Egi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ego = Bgo ^ __andn_u64(Bgu, Bga);
    Egu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(((unsigned long long*)input)[1] ^ De, 1);
    Bke = ROL64(((unsigned long long*)input)[7] ^ Di, 6);
    Bki = ROL64(Do, 25);
    Bko = ROL64(Du, 8);
    Bku = ROL64(Da ^ 0x8000000000000000, 18);
    Eka = Bka ^ __andn_u64(Bke, Bki);
    Eke = Bke ^ __andn_u64(Bki, Bko);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Eko = Bko ^ __andn_u64(Bku, Bka);
    Eku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(((unsigned long long*)input)[4] ^ Du, 27);
    Bme = ROL64(((unsigned long long*)input)[5] ^ Da, 36);
    Bmi = ROL64(De, 10);
    Bmo = ROL64(Di, 15);
    Bmu = ROL64(Do, 56);
    Ema = Bma ^ __andn_u64(Bme, Bmi);
    Eme = Bme ^ __andn_u64(Bmi, Bmo);
    Emi = Bmi ^ __andn_u64(Bmo, Bmu);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Emu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(((unsigned long long*)input)[2] ^ Di, 62);
    Bse = ROL64(Do ^ 0x0700, 55);
    Bsi = ROL64(Du, 39);
    Bso = ROL64(Da, 41);
    Bsu = ROL64(De, 2);
    Esa = Bsa ^ __andn_u64(Bse, Bsi);
    Ese = Bse ^ __andn_u64(Bsi, Bso);
    Esi = Bsi ^ __andn_u64(Bso, Bsu);
    Eso = Bso ^ __andn_u64(Bsu, Bsa);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    Ce = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    Ci = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    Co = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    Cu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Eba ^= Da;
    Bbe = ROL64(Ege ^ De, 44);
    Bbi = ROL64(Eki ^ Di, 43);
    Bbo = ROL64(Emo ^ Do, 21);
    Bbu = ROL64(Esu ^ Du, 14);
    Aba = Eba ^ __andn_u64(Bbe, Bbi) ^ 0x800000000000008bULL;
    Abe = Bbe ^ __andn_u64(Bbi, Bbo);
    Abi = Bbi ^ __andn_u64(Bbo, Bbu);
    Abo = Bbo ^ __andn_u64(Bbu, Eba);
    Abu = Bbu ^ __andn_u64(Eba, Bbe);
    Bga = ROL64(Ebo ^ Do, 28);
    Bge = ROL64(Egu ^ Du, 20);
    Bgi = ROL64(Eka ^ Da, 3);
    Bgo = ROL64(Eme ^ De, 45);
    Bgu = ROL64(Esi ^ Di, 61);
    Aga = Bga ^ __andn_u64(Bge, Bgi);
    Age = Bge ^ __andn_u64(Bgi, Bgo);
    Agi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ago = Bgo ^ __andn_u64(Bgu, Bga);
    Agu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Ebe ^ De, 1);
    Bke = ROL64(Egi ^ Di, 6);
    Bki = ROL64(Eko ^ Do, 25);
    Bko = ROL64(Emu ^ Du, 8);
    Bku = ROL64(Esa ^ Da, 18);
    Aka = Bka ^ __andn_u64(Bke, Bki);
    Ake = Bke ^ __andn_u64(Bki, Bko);
    Aki = Bki ^ __andn_u64(Bko, Bku);
    Ako = Bko ^ __andn_u64(Bku, Bka);
    Aku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Ebu ^ Du, 27);
    Bme = ROL64(Ega ^ Da, 36);
    Bmi = ROL64(Eke ^ De, 10);
    Bmo = ROL64(Emi ^ Di, 15);
    Bmu = ROL64(Eso ^ Do, 56);
    Ama = Bma ^ __andn_u64(Bme, Bmi);
    Ame = Bme ^ __andn_u64(Bmi, Bmo);
    Ami = Bmi ^ __andn_u64(Bmo, Bmu);
    Amo = Bmo ^ __andn_u64(Bmu, Bma);
    Amu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Ebi ^ Di, 62);
    Bse = ROL64(Ego ^ Do, 55);
    Bsi = ROL64(Eku ^ Du, 39);
    Bso = ROL64(Ema ^ Da, 41);
    Bsu = ROL64(Ese ^ De, 2);
    Asa = Bsa ^ __andn_u64(Bse, Bsi);
    Ase = Bse ^ __andn_u64(Bsi, Bso);
    Asi = Bsi ^ __andn_u64(Bso, Bsu);
    Aso = Bso ^ __andn_u64(Bsu, Bsa);
    Asu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Aba ^= Da;
    Bbe = ROL64(Age ^ De, 44);
    Bbi = ROL64(Aki ^ Di, 43);
    Bbo = ROL64(Amo ^ Do, 21);
    Bbu = ROL64(Asu ^ Du, 14);
    Eba = Aba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000000008089ULL;
    Ebe = Bbe ^ __andn_u64(Bbi, Bbo);
    Ebi = Bbi ^ __andn_u64(Bbo, Bbu);
    Ebo = Bbo ^ __andn_u64(Bbu, Aba);
    Ebu = Bbu ^ __andn_u64(Aba, Bbe);
    Bga = ROL64(Abo ^ Do, 28);
    Bge = ROL64(Agu ^ Du, 20);
    Bgi = ROL64(Aka ^ Da, 3);
    Bgo = ROL64(Ame ^ De, 45);
    Bgu = ROL64(Asi ^ Di, 61);
    Ega = Bga ^ __andn_u64(Bge, Bgi);
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Egi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ego = Bgo ^ __andn_u64(Bgu, Bga);
    Egu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Abe ^ De, 1);
    Bke = ROL64(Agi ^ Di, 6);
    Bki = ROL64(Ako ^ Do, 25);
    Bko = ROL64(Amu ^ Du, 8);
    Bku = ROL64(Asa ^ Da, 18);
    Eka = Bka ^ __andn_u64(Bke, Bki);
    Eke = Bke ^ __andn_u64(Bki, Bko);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Eko = Bko ^ __andn_u64(Bku, Bka);
    Eku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Abu ^ Du, 27);
    Bme = ROL64(Aga ^ Da, 36);
    Bmi = ROL64(Ake ^ De, 10);
    Bmo = ROL64(Ami ^ Di, 15);
    Bmu = ROL64(Aso ^ Do, 56);
    Ema = Bma ^ __andn_u64(Bme, Bmi);
    Eme = Bme ^ __andn_u64(Bmi, Bmo);
    Emi = Bmi ^ __andn_u64(Bmo, Bmu);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Emu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Abi ^ Di, 62);
    Bse = ROL64(Ago ^ Do, 55);
    Bsi = ROL64(Aku ^ Du, 39);
    Bso = ROL64(Ama ^ Da, 41);
    Bsu = ROL64(Ase ^ De, 2);
    Esa = Bsa ^ __andn_u64(Bse, Bsi);
    Ese = Bse ^ __andn_u64(Bsi, Bso);
    Esi = Bsi ^ __andn_u64(Bso, Bsu);
    Eso = Bso ^ __andn_u64(Bsu, Bsa);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    Ce = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    Ci = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    Co = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    Cu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Eba ^= Da;
    Bbe = ROL64(Ege ^ De, 44);
    Bbi = ROL64(Eki ^ Di, 43);
    Bbo = ROL64(Emo ^ Do, 21);
    Bbu = ROL64(Esu ^ Du, 14);
    Aba = Eba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000000008003ULL;
    Abe = Bbe ^ __andn_u64(Bbi, Bbo);
    Abi = Bbi ^ __andn_u64(Bbo, Bbu);
    Abo = Bbo ^ __andn_u64(Bbu, Eba);
    Abu = Bbu ^ __andn_u64(Eba, Bbe);
    Bga = ROL64(Ebo ^ Do, 28);
    Bge = ROL64(Egu ^ Du, 20);
    Bgi = ROL64(Eka ^ Da, 3);
    Bgo = ROL64(Eme ^ De, 45);
    Bgu = ROL64(Esi ^ Di, 61);
    Aga = Bga ^ __andn_u64(Bge, Bgi);
    Age = Bge ^ __andn_u64(Bgi, Bgo);
    Agi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ago = Bgo ^ __andn_u64(Bgu, Bga);
    Agu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Ebe ^ De, 1);
    Bke = ROL64(Egi ^ Di, 6);
    Bki = ROL64(Eko ^ Do, 25);
    Bko = ROL64(Emu ^ Du, 8);
    Bku = ROL64(Esa ^ Da, 18);
    Aka = Bka ^ __andn_u64(Bke, Bki);
    Ake = Bke ^ __andn_u64(Bki, Bko);
    Aki = Bki ^ __andn_u64(Bko, Bku);
    Ako = Bko ^ __andn_u64(Bku, Bka);
    Aku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Ebu ^ Du, 27);
    Bme = ROL64(Ega ^ Da, 36);
    Bmi = ROL64(Eke ^ De, 10);
    Bmo = ROL64(Emi ^ Di, 15);
    Bmu = ROL64(Eso ^ Do, 56);
    Ama = Bma ^ __andn_u64(Bme, Bmi);
    Ame = Bme ^ __andn_u64(Bmi, Bmo);
    Ami = Bmi ^ __andn_u64(Bmo, Bmu);
    Amo = Bmo ^ __andn_u64(Bmu, Bma);
    Amu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Ebi ^ Di, 62);
    Bse = ROL64(Ego ^ Do, 55);
    Bsi = ROL64(Eku ^ Du, 39);
    Bso = ROL64(Ema ^ Da, 41);
    Bsu = ROL64(Ese ^ De, 2);
    Asa = Bsa ^ __andn_u64(Bse, Bsi);
    Ase = Bse ^ __andn_u64(Bsi, Bso);
    Asi = Bsi ^ __andn_u64(Bso, Bsu);
    Aso = Bso ^ __andn_u64(Bsu, Bsa);
    Asu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Aba ^= Da;
    Bbe = ROL64(Age ^ De, 44);
    Bbi = ROL64(Aki ^ Di, 43);
    Bbo = ROL64(Amo ^ Do, 21);
    Bbu = ROL64(Asu ^ Du, 14);
    Eba = Aba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000000008002ULL;
    Ebe = Bbe ^ __andn_u64(Bbi, Bbo);
    Ebi = Bbi ^ __andn_u64(Bbo, Bbu);
    Ebo = Bbo ^ __andn_u64(Bbu, Aba);
    Ebu = Bbu ^ __andn_u64(Aba, Bbe);
    Bga = ROL64(Abo ^ Do, 28);
    Bge = ROL64(Agu ^ Du, 20);
    Bgi = ROL64(Aka ^ Da, 3);
    Bgo = ROL64(Ame ^ De, 45);
    Bgu = ROL64(Asi ^ Di, 61);
    Ega = Bga ^ __andn_u64(Bge, Bgi);
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Egi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ego = Bgo ^ __andn_u64(Bgu, Bga);
    Egu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Abe ^ De, 1);
    Bke = ROL64(Agi ^ Di, 6);
    Bki = ROL64(Ako ^ Do, 25);
    Bko = ROL64(Amu ^ Du, 8);
    Bku = ROL64(Asa ^ Da, 18);
    Eka = Bka ^ __andn_u64(Bke, Bki);
    Eke = Bke ^ __andn_u64(Bki, Bko);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Eko = Bko ^ __andn_u64(Bku, Bka);
    Eku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Abu ^ Du, 27);
    Bme = ROL64(Aga ^ Da, 36);
    Bmi = ROL64(Ake ^ De, 10);
    Bmo = ROL64(Ami ^ Di, 15);
    Bmu = ROL64(Aso ^ Do, 56);
    Ema = Bma ^ __andn_u64(Bme, Bmi);
    Eme = Bme ^ __andn_u64(Bmi, Bmo);
    Emi = Bmi ^ __andn_u64(Bmo, Bmu);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Emu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Abi ^ Di, 62);
    Bse = ROL64(Ago ^ Do, 55);
    Bsi = ROL64(Aku ^ Du, 39);
    Bso = ROL64(Ama ^ Da, 41);
    Bsu = ROL64(Ase ^ De, 2);
    Esa = Bsa ^ __andn_u64(Bse, Bsi);
    Ese = Bse ^ __andn_u64(Bsi, Bso);
    Esi = Bsi ^ __andn_u64(Bso, Bsu);
    Eso = Bso ^ __andn_u64(Bsu, Bsa);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    Ce = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    Ci = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    Co = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    Cu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Eba ^= Da;
    Bbe = ROL64(Ege ^ De, 44);
    Bbi = ROL64(Eki ^ Di, 43);
    Bbo = ROL64(Emo ^ Do, 21);
    Bbu = ROL64(Esu ^ Du, 14);
    Aba = Eba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000000000080ULL;
    Abe = Bbe ^ __andn_u64(Bbi, Bbo);
    Abi = Bbi ^ __andn_u64(Bbo, Bbu);
    Abo = Bbo ^ __andn_u64(Bbu, Eba);
    Abu = Bbu ^ __andn_u64(Eba, Bbe);
    Bga = ROL64(Ebo ^ Do, 28);
    Bge = ROL64(Egu ^ Du, 20);
    Bgi = ROL64(Eka ^ Da, 3);
    Bgo = ROL64(Eme ^ De, 45);
    Bgu = ROL64(Esi ^ Di, 61);
    Aga = Bga ^ __andn_u64(Bge, Bgi);
    Age = Bge ^ __andn_u64(Bgi, Bgo);
    Agi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ago = Bgo ^ __andn_u64(Bgu, Bga);
    Agu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Ebe ^ De, 1);
    Bke = ROL64(Egi ^ Di, 6);
    Bki = ROL64(Eko ^ Do, 25);
    Bko = ROL64(Emu ^ Du, 8);
    Bku = ROL64(Esa ^ Da, 18);
    Aka = Bka ^ __andn_u64(Bke, Bki);
    Ake = Bke ^ __andn_u64(Bki, Bko);
    Aki = Bki ^ __andn_u64(Bko, Bku);
    Ako = Bko ^ __andn_u64(Bku, Bka);
    Aku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Ebu ^ Du, 27);
    Bme = ROL64(Ega ^ Da, 36);
    Bmi = ROL64(Eke ^ De, 10);
    Bmo = ROL64(Emi ^ Di, 15);
    Bmu = ROL64(Eso ^ Do, 56);
    Ama = Bma ^ __andn_u64(Bme, Bmi);
    Ame = Bme ^ __andn_u64(Bmi, Bmo);
    Ami = Bmi ^ __andn_u64(Bmo, Bmu);
    Amo = Bmo ^ __andn_u64(Bmu, Bma);
    Amu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Ebi ^ Di, 62);
    Bse = ROL64(Ego ^ Do, 55);
    Bsi = ROL64(Eku ^ Du, 39);
    Bso = ROL64(Ema ^ Da, 41);
    Bsu = ROL64(Ese ^ De, 2);
    Asa = Bsa ^ __andn_u64(Bse, Bsi);
    Ase = Bse ^ __andn_u64(Bsi, Bso);
    Asi = Bsi ^ __andn_u64(Bso, Bsu);
    Aso = Bso ^ __andn_u64(Bsu, Bsa);
    Asu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Aba ^= Da;
    Bbe = ROL64(Age ^ De, 44);
    Bbi = ROL64(Aki ^ Di, 43);
    Bbo = ROL64(Amo ^ Do, 21);
    Bbu = ROL64(Asu ^ Du, 14);
    Eba = Aba ^ __andn_u64(Bbe, Bbi) ^ 0x000000000000800aULL;
    Ebe = Bbe ^ __andn_u64(Bbi, Bbo);
    Ebi = Bbi ^ __andn_u64(Bbo, Bbu);
    Ebo = Bbo ^ __andn_u64(Bbu, Aba);
    Ebu = Bbu ^ __andn_u64(Aba, Bbe);
    Bga = ROL64(Abo ^ Do, 28);
    Bge = ROL64(Agu ^ Du, 20);
    Bgi = ROL64(Aka ^ Da, 3);
    Bgo = ROL64(Ame ^ De, 45);
    Bgu = ROL64(Asi ^ Di, 61);
    Ega = Bga ^ __andn_u64(Bge, Bgi);
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Egi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ego = Bgo ^ __andn_u64(Bgu, Bga);
    Egu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Abe ^ De, 1);
    Bke = ROL64(Agi ^ Di, 6);
    Bki = ROL64(Ako ^ Do, 25);
    Bko = ROL64(Amu ^ Du, 8);
    Bku = ROL64(Asa ^ Da, 18);
    Eka = Bka ^ __andn_u64(Bke, Bki);
    Eke = Bke ^ __andn_u64(Bki, Bko);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Eko = Bko ^ __andn_u64(Bku, Bka);
    Eku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Abu ^ Du, 27);
    Bme = ROL64(Aga ^ Da, 36);
    Bmi = ROL64(Ake ^ De, 10);
    Bmo = ROL64(Ami ^ Di, 15);
    Bmu = ROL64(Aso ^ Do, 56);
    Ema = Bma ^ __andn_u64(Bme, Bmi);
    Eme = Bme ^ __andn_u64(Bmi, Bmo);
    Emi = Bmi ^ __andn_u64(Bmo, Bmu);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Emu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Abi ^ Di, 62);
    Bse = ROL64(Ago ^ Do, 55);
    Bsi = ROL64(Aku ^ Du, 39);
    Bso = ROL64(Ama ^ Da, 41);
    Bsu = ROL64(Ase ^ De, 2);
    Esa = Bsa ^ __andn_u64(Bse, Bsi);
    Ese = Bse ^ __andn_u64(Bsi, Bso);
    Esi = Bsi ^ __andn_u64(Bso, Bsu);
    Eso = Bso ^ __andn_u64(Bsu, Bsa);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    Ce = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    Ci = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    Co = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    Cu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Eba ^= Da;
    Bbe = ROL64(Ege ^ De, 44);
    Bbi = ROL64(Eki ^ Di, 43);
    Bbo = ROL64(Emo ^ Do, 21);
    Bbu = ROL64(Esu ^ Du, 14);
    Aba = Eba ^ __andn_u64(Bbe, Bbi) ^ 0x800000008000000aULL;
    Abe = Bbe ^ __andn_u64(Bbi, Bbo);
    Abi = Bbi ^ __andn_u64(Bbo, Bbu);
    Abo = Bbo ^ __andn_u64(Bbu, Eba);
    Abu = Bbu ^ __andn_u64(Eba, Bbe);
    Bga = ROL64(Ebo ^ Do, 28);
    Bge = ROL64(Egu ^ Du, 20);
    Bgi = ROL64(Eka ^ Da, 3);
    Bgo = ROL64(Eme ^ De, 45);
    Bgu = ROL64(Esi ^ Di, 61);
    Aga = Bga ^ __andn_u64(Bge, Bgi);
    Age = Bge ^ __andn_u64(Bgi, Bgo);
    Agi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ago = Bgo ^ __andn_u64(Bgu, Bga);
    Agu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Ebe ^ De, 1);
    Bke = ROL64(Egi ^ Di, 6);
    Bki = ROL64(Eko ^ Do, 25);
    Bko = ROL64(Emu ^ Du, 8);
    Bku = ROL64(Esa ^ Da, 18);
    Aka = Bka ^ __andn_u64(Bke, Bki);
    Ake = Bke ^ __andn_u64(Bki, Bko);
    Aki = Bki ^ __andn_u64(Bko, Bku);
    Ako = Bko ^ __andn_u64(Bku, Bka);
    Aku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Ebu ^ Du, 27);
    Bme = ROL64(Ega ^ Da, 36);
    Bmi = ROL64(Eke ^ De, 10);
    Bmo = ROL64(Emi ^ Di, 15);
    Bmu = ROL64(Eso ^ Do, 56);
    Ama = Bma ^ __andn_u64(Bme, Bmi);
    Ame = Bme ^ __andn_u64(Bmi, Bmo);
    Ami = Bmi ^ __andn_u64(Bmo, Bmu);
    Amo = Bmo ^ __andn_u64(Bmu, Bma);
    Amu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Ebi ^ Di, 62);
    Bse = ROL64(Ego ^ Do, 55);
    Bsi = ROL64(Eku ^ Du, 39);
    Bso = ROL64(Ema ^ Da, 41);
    Bsu = ROL64(Ese ^ De, 2);
    Asa = Bsa ^ __andn_u64(Bse, Bsi);
    Ase = Bse ^ __andn_u64(Bsi, Bso);
    Asi = Bsi ^ __andn_u64(Bso, Bsu);
    Aso = Bso ^ __andn_u64(Bsu, Bsa);
    Asu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Aba ^= Da;
    Bbe = ROL64(Age ^ De, 44);
    Bbi = ROL64(Aki ^ Di, 43);
    Bbo = ROL64(Amo ^ Do, 21);
    Bbu = ROL64(Asu ^ Du, 14);
    Eba = Aba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000080008081ULL;
    Ebe = Bbe ^ __andn_u64(Bbi, Bbo);
    Ebi = Bbi ^ __andn_u64(Bbo, Bbu);
    Ebo = Bbo ^ __andn_u64(Bbu, Aba);
    Ebu = Bbu ^ __andn_u64(Aba, Bbe);
    Bga = ROL64(Abo ^ Do, 28);
    Bge = ROL64(Agu ^ Du, 20);
    Bgi = ROL64(Aka ^ Da, 3);
    Bgo = ROL64(Ame ^ De, 45);
    Bgu = ROL64(Asi ^ Di, 61);
    Ega = Bga ^ __andn_u64(Bge, Bgi);
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Egi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ego = Bgo ^ __andn_u64(Bgu, Bga);
    Egu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Abe ^ De, 1);
    Bke = ROL64(Agi ^ Di, 6);
    Bki = ROL64(Ako ^ Do, 25);
    Bko = ROL64(Amu ^ Du, 8);
    Bku = ROL64(Asa ^ Da, 18);
    Eka = Bka ^ __andn_u64(Bke, Bki);
    Eke = Bke ^ __andn_u64(Bki, Bko);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Eko = Bko ^ __andn_u64(Bku, Bka);
    Eku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Abu ^ Du, 27);
    Bme = ROL64(Aga ^ Da, 36);
    Bmi = ROL64(Ake ^ De, 10);
    Bmo = ROL64(Ami ^ Di, 15);
    Bmu = ROL64(Aso ^ Do, 56);
    Ema = Bma ^ __andn_u64(Bme, Bmi);
    Eme = Bme ^ __andn_u64(Bmi, Bmo);
    Emi = Bmi ^ __andn_u64(Bmo, Bmu);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Emu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Abi ^ Di, 62);
    Bse = ROL64(Ago ^ Do, 55);
    Bsi = ROL64(Aku ^ Du, 39);
    Bso = ROL64(Ama ^ Da, 41);
    Bsu = ROL64(Ase ^ De, 2);
    Esa = Bsa ^ __andn_u64(Bse, Bsi);
    Ese = Bse ^ __andn_u64(Bsi, Bso);
    Esi = Bsi ^ __andn_u64(Bso, Bsu);
    Eso = Bso ^ __andn_u64(Bsu, Bsa);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
    Ce = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
    Ci = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
    Co = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
    Cu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Eba ^= Da;
    Bbe = ROL64(Ege ^ De, 44);
    Bbi = ROL64(Eki ^ Di, 43);
    Bbo = ROL64(Emo ^ Do, 21);
    Bbu = ROL64(Esu ^ Du, 14);
    Aba = Eba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000000008080ULL;
    Abe = Bbe ^ __andn_u64(Bbi, Bbo);
    Abi = Bbi ^ __andn_u64(Bbo, Bbu);
    Abo = Bbo ^ __andn_u64(Bbu, Eba);
    Abu = Bbu ^ __andn_u64(Eba, Bbe);
    Bga = ROL64(Ebo ^ Do, 28);
    Bge = ROL64(Egu ^ Du, 20);
    Bgi = ROL64(Eka ^ Da, 3);
    Bgo = ROL64(Eme ^ De, 45);
    Bgu = ROL64(Esi ^ Di, 61);
    Aga = Bga ^ __andn_u64(Bge, Bgi);
    Age = Bge ^ __andn_u64(Bgi, Bgo);
    Agi = Bgi ^ __andn_u64(Bgo, Bgu);
    Ago = Bgo ^ __andn_u64(Bgu, Bga);
    Agu = Bgu ^ __andn_u64(Bga, Bge);
    Bka = ROL64(Ebe ^ De, 1);
    Bke = ROL64(Egi ^ Di, 6);
    Bki = ROL64(Eko ^ Do, 25);
    Bko = ROL64(Emu ^ Du, 8);
    Bku = ROL64(Esa ^ Da, 18);
    Aka = Bka ^ __andn_u64(Bke, Bki);
    Ake = Bke ^ __andn_u64(Bki, Bko);
    Aki = Bki ^ __andn_u64(Bko, Bku);
    Ako = Bko ^ __andn_u64(Bku, Bka);
    Aku = Bku ^ __andn_u64(Bka, Bke);
    Bma = ROL64(Ebu ^ Du, 27);
    Bme = ROL64(Ega ^ Da, 36);
    Bmi = ROL64(Eke ^ De, 10);
    Bmo = ROL64(Emi ^ Di, 15);
    Bmu = ROL64(Eso ^ Do, 56);
    Ama = Bma ^ __andn_u64(Bme, Bmi);
    Ame = Bme ^ __andn_u64(Bmi, Bmo);
    Ami = Bmi ^ __andn_u64(Bmo, Bmu);
    Amo = Bmo ^ __andn_u64(Bmu, Bma);
    Amu = Bmu ^ __andn_u64(Bma, Bme);
    Bsa = ROL64(Ebi ^ Di, 62);
    Bse = ROL64(Ego ^ Do, 55);
    Bsi = ROL64(Eku ^ Du, 39);
    Bso = ROL64(Ema ^ Da, 41);
    Bsu = ROL64(Ese ^ De, 2);
    Asa = Bsa ^ __andn_u64(Bse, Bsi);
    Ase = Bse ^ __andn_u64(Bsi, Bso);
    Asi = Bsi ^ __andn_u64(Bso, Bsu);
    Aso = Bso ^ __andn_u64(Bsu, Bsa);
    Asu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
    Ce = Abe ^ Age ^ Ake ^ Ame ^ Ase;
    Ci = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
    Co = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
    Cu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

    Da = Cu ^ ROL64(Ce, 1);
    De = Ca ^ ROL64(Ci, 1);
    Di = Ce ^ ROL64(Co, 1);
    Do = Ci ^ ROL64(Cu, 1);
    Du = Co ^ ROL64(Ca, 1);
    Bba = Aba ^ Da;
    Bbe = ROL64(Age ^ De, 44);
    Bbi = ROL64(Aki ^ Di, 43);
    Bbo = ROL64(Amo ^ Do, 21);
    Bbu = ROL64(Asu ^ Du, 14);
    Bga = ROL64(Abo ^ Do, 28);
    Bge = ROL64(Agu ^ Du, 20);
    Bgi = ROL64(Aka ^ Da, 3);
    Bgo = ROL64(Ame ^ De, 45);
    Bgu = ROL64(Asi ^ Di, 61);
    Bka = ROL64(Abe ^ De, 1);
    Bke = ROL64(Agi ^ Di, 6);
    Bki = ROL64(Ako ^ Do, 25);
    Bko = ROL64(Amu ^ Du, 8);
    Bku = ROL64(Asa ^ Da, 18);
    Bma = ROL64(Abu ^ Du, 27);
    Bme = ROL64(Aga ^ Da, 36);
    Bmi = ROL64(Ake ^ De, 10);
    Bmo = ROL64(Ami ^ Di, 15);
    Bmu = ROL64(Aso ^ Do, 56);
    Bsa = ROL64(Abi ^ Di, 62);
    Bse = ROL64(Ago ^ Do, 55);
    Bsi = ROL64(Aku ^ Du, 39);
    Bso = ROL64(Ama ^ Da, 41);
    Bsu = ROL64(Ase ^ De, 2);
    Eba = Bba ^ __andn_u64(Bbe, Bbi) ^ 0x0000000080000001ULL;
    Ege = Bge ^ __andn_u64(Bgi, Bgo);
    Eki = Bki ^ __andn_u64(Bko, Bku);
    Emo = Bmo ^ __andn_u64(Bmu, Bma);
    Esu = Bsu ^ __andn_u64(Bsa, Bse);
    Ca = Eba ^ Bga ^ Bka ^ Bma ^ Bsa ^ __andn_u64(Bge, Bgi) ^ __andn_u64(Bke, Bki) ^ __andn_u64(Bme, Bmi) ^ __andn_u64(Bse, Bsi);
    Ce = Bbe ^ Ege ^ Bke ^ Bme ^ Bse ^ __andn_u64(Bbi, Bbo) ^ __andn_u64(Bki, Bko) ^ __andn_u64(Bmi, Bmo) ^ __andn_u64(Bsi, Bso);
    Ci = Bbi ^ Bgi ^ Eki ^ Bmi ^ Bsi ^ __andn_u64(Bbo, Bbu) ^ __andn_u64(Bgo, Bgu) ^ __andn_u64(Bmo, Bmu) ^ __andn_u64(Bso, Bsu);
    Co = Bbo ^ Bgo ^ Bko ^ Emo ^ Bso ^ __andn_u64(Bbu, Bba) ^ __andn_u64(Bgu, Bga) ^ __andn_u64(Bku, Bka) ^ __andn_u64(Bsu, Bsa);
    Cu = Bbu ^ Bgu ^ Bku ^ Bmu ^ Esu ^ __andn_u64(Bba, Bbe) ^ __andn_u64(Bga, Bge) ^ __andn_u64(Bka, Bke) ^ __andn_u64(Bma, Bme);

    Bba = Eba ^ Cu ^ ROL64(Ce, 1);
    Bbe = ROL64(Ege ^ Ca ^ ROL64(Ci, 1), 44);
    Bbi = ROL64(Eki ^ Ce ^ ROL64(Co, 1), 43);
    Bbo = ROL64(Emo ^ Ci ^ ROL64(Cu, 1), 21);
    Bbu = ROL64(Esu ^ Co ^ ROL64(Ca, 1), 14);
    ((unsigned long long*)output)[0] = Bba ^ __andn_u64(Bbe, Bbi) ^ 0x8000000080008008ULL;
    ((unsigned long long*)output)[1] = Bbe ^ __andn_u64(Bbi, Bbo);
    ((unsigned long long*)output)[2] = Bbi ^ __andn_u64(Bbo, Bbu);
    ((unsigned long long*)output)[3] = Bbo ^ __andn_u64(Bbu, Bba);
#endif
}

void random(unsigned char *publicKey, unsigned char *nonce, unsigned char *output, unsigned int outputSize)
{
    unsigned char state[200] __attribute__((aligned(32)));
    *((__m256i *)&state[0]) = *((__m256i *)publicKey);
    //memcpy(state, publicKey, 32);
    *((__m256i *)&state[32]) = *((__m256i *)nonce);
    //memcpy(state+32, nonce, 32);
    memset(&state[64], 0, sizeof(state) - 64);
    for (unsigned int i = 0; i < outputSize / sizeof(state); i++)
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, sizeof(state));
        output += sizeof(state);
    }
    if (outputSize % sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, outputSize % sizeof(state));
    }
}

struct RequestResponseHeader
{
private:
    unsigned char _size[3];
    unsigned char _protocol;
    unsigned char _dejavu[3];
    unsigned char _type;

public:
    inline unsigned int size()
    {
        return (*((unsigned int*)_size)) & 0xFFFFFF;
    }

    inline void setSize(unsigned int size)
    {
        _size[0] = (unsigned char)size;
        _size[1] = (unsigned char)(size >> 8);
        _size[2] = (unsigned char)(size >> 16);
    }

    inline unsigned char protocol()
    {
        return _protocol;
    }

    inline void setProtocol()
    {
        _protocol = 0;
    }

    inline bool isDejavuZero()
    {
        return !(_dejavu[0] | _dejavu[1] | _dejavu[2]);
    }

    inline void zeroDejavu()
    {
        _dejavu[0] = 0;
        _dejavu[1] = 0;
        _dejavu[2] = 0;
    }

    inline void randomizeDejavu()
    {
        unsigned int random;
        _rdrand32_step(&random);
        if (!random)
        {
            random = 1;
        }
        _dejavu[0] = (unsigned char)random;
        _dejavu[1] = (unsigned char)(random >> 8);
        _dejavu[2] = (unsigned char)(random >> 16);
    }

    inline unsigned char type()
    {
        return _type;
    }

    inline void setType(const unsigned char type)
    {
        _type = type;
    }
};

#define BROADCAST_MESSAGE 1

typedef struct
{
    unsigned char sourcePublicKey[32];
    unsigned char destinationPublicKey[32];
    unsigned char gammingNonce[32];
} Message;

__m256i _mm256_mullo_epi8(const __m256i a, const __m256i b) {
    // unpack and multiply
    const __m256i dst_even = _mm256_mullo_epi16(a, b);
    __m256i dst_odd = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8),_mm256_srli_epi16(b, 8));
    return _mm256_or_si256(_mm256_slli_epi16(dst_odd, 8), _mm256_and_si256(dst_even, _mm256_set1_epi16(0xFF)));
}


constexpr const unsigned short gcVecBytes = 32;
const __m256i gPermuteAllSigns = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
const __m256i vInvDiv3_8bit = _mm256_set1_epi16(171);
const __m256i v3_16bit = _mm256_set1_epi16(3);
const __m256i v1_8bit = _mm256_set1_epi8(1);
const __m256i vBitOnes = _mm256_set1_epi8(0xff);
const static __m256i ZERO = _mm256_setzero_si256();
bool gVerifySolution = false;

struct Miner
{
#define DATA_LENGTH 2000
#define INFO_LENGTH 1000
#define NUMBER_OF_INPUT_NEURONS 1000
#define NUMBER_OF_OUTPUT_NEURONS 1000
#define MAX_INPUT_DURATION 20
#define MAX_OUTPUT_DURATION 20
#define SOLUTION_THRESHOLD 1130

    // unsigned long long data[DATA_LENGTH / 64];
    int data[DATA_LENGTH];
    alignas(32) unsigned char computorPublicKey[32];

    void initialize()
    {
        alignas(32) unsigned char randomSeed[32];
        memset(randomSeed, 0, sizeof(randomSeed));
        randomSeed[0] = 99;
        randomSeed[1] = 23;
        randomSeed[2] = 147;
        randomSeed[3] = 2;
        randomSeed[4] = 202;
        randomSeed[5] = 0;
        randomSeed[6] = 0;
        randomSeed[7] = 0;
        random(randomSeed, randomSeed, (unsigned char*)data, sizeof(data));

        memset(computorPublicKey, 0, sizeof(computorPublicKey));
    }

    void getComputorPublicKey(unsigned char computorPublicKey[32])
    {
        *((__m256i*)computorPublicKey) = *((__m256i*)this->computorPublicKey);
    }

    void setComputorPublicKey(unsigned char computorPublicKey[32])
    {
        *((__m256i*)this->computorPublicKey) = *((__m256i*)computorPublicKey);
    }

    static void populateSynapses(const char* __restrict synapses, const unsigned int nSynapses, uint8_t* __restrict pConSyns) {
      assert(nSynapses % gcVecBytes == 0);
      const int nVect = nSynapses / gcVecBytes;
      #pragma GCC unroll 4
      for(int i=0; i<nVect; i++) {
        const __m256i orig = _mm256_load_si256(reinterpret_cast<const __m256i*>(synapses+i*gcVecBytes));
        const __m256i a = _mm256_unpacklo_epi8(orig, ZERO);
        const __m256i b = _mm256_unpackhi_epi8(orig, ZERO);
        const __m256i origMinus1 = _mm256_sub_epi8(orig, v1_8bit);
        const __m256i amul = _mm256_mullo_epi16(a, vInvDiv3_8bit);
        const __m256i bmul = _mm256_mullo_epi16(b, vInvDiv3_8bit);
        const __m256i asr = _mm256_srli_epi16(amul, 9);
        const __m256i bsr = _mm256_srli_epi16(bmul, 9);
        const __m256i afull = _mm256_mullo_epi16(asr, v3_16bit);
        const __m256i bfull = _mm256_mullo_epi16(bsr, v3_16bit);
        const __m256i allFull = _mm256_packus_epi16(afull, bfull);
        const __m256i target = _mm256_sub_epi8(origMinus1, allFull);

        //_mm256_store_si256(reinterpret_cast<__m256i*>(synapses+i*gcVecBytes), target);
        const __m256i vBits0 = _mm256_slli_epi16(target, 7);
        const int bits1 = _mm256_movemask_epi8(target);
        const int bits0 = _mm256_movemask_epi8(vBits0);
        const uint64_t withHigher = _pdep_u64(bits1, 0xAAAAAAAAAAAAAAAA);
        const uint64_t withLower =  _pdep_u64(bits0, 0x5555555555555555);
        *reinterpret_cast<uint64_t* __restrict>(pConSyns+(i*gcVecBytes/4)) = withHigher | withLower;
      }
    }

    static __m256i extractConSyns(const uint64_t condensed) {
      const uint64_t qword0 = _pdep_u64(condensed, 0xc0c0c0c0c0c0c0c0);
      const uint64_t qword1 = _pdep_u64(condensed>>16, 0xc0c0c0c0c0c0c0c0);
      const uint64_t qword2 = _pdep_u64(condensed>>32, 0xc0c0c0c0c0c0c0c0);
      const uint64_t qword3 = _pdep_u64(condensed>>48, 0xc0c0c0c0c0c0c0c0);
      const __m256i highBits = _mm256_setr_epi64x(qword0, qword1, qword2, qword3);
      const __m256i lowBits = _mm256_srli_epi16(highBits, 6);
      const __m256i positioned = _mm256_blendv_epi8(lowBits, vBitOnes, highBits);
      return positioned;
    }

    static __m256i extractConSyns(const void* __restrict buffer) {
      const uint64_t condensed = *reinterpret_cast<const uint64_t*>(buffer);
      return extractConSyns(condensed);
    }

    static void fireNeuronsVect(int* __restrict source_end, const uint8_t* __restrict conSyns, int nNeurons,
      const uint8_t* conNeurs)
    {
      // Handle separately 128, 64, 32, 16 and 8 bit batches
      __m256i total0 = ZERO, total1 = ZERO, total2 = ZERO, total3 = ZERO;
      int modifier = -nNeurons;
      const __m256i allBitPairs01 = _mm256_set1_epi8(0x55);
      const __m256i allBytes3 = _mm256_set1_epi8(3);
      const int nVects128 = nNeurons / 128;
      if(nVects128 >= 1) {
        for(int i=0; i<nVects128; i++) {
          const __m256i neurons = _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(conNeurs)+i);
          const __m256i synapses = _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(conSyns)+i);
          const __m256i vXorred = _mm256_xor_si256(neurons, synapses);
          const __m256i synNonzero = _mm256_and_si256(synapses, allBitPairs01);
          const __m256i shifted = _mm256_slli_epi64(synNonzero, 1);
          const __m256i mask = _mm256_or_si256(shifted, allBitPairs01);
          const __m256i increments = _mm256_and_si256(vXorred, mask);
          // Now |increments| contain 128 bit pairs, each being -1, 0, or +1

          const __m256i shift2 = _mm256_srli_epi64(increments, 2);
          const __m256i shift4 = _mm256_srli_epi64(increments, 4);
          const __m256i shift6 = _mm256_srli_epi64(increments, 6);

          const __m256i counts0 = _mm256_and_si256(increments, allBytes3);
          const __m256i counts1 = _mm256_and_si256(shift2, allBytes3);
          const __m256i counts2 = _mm256_and_si256(shift4, allBytes3);
          const __m256i counts3 = _mm256_and_si256(shift6, allBytes3);

          total0 = _mm256_add_epi8(total0, counts0);
          total1 = _mm256_add_epi8(total1, counts1);
          total2 = _mm256_add_epi8(total2, counts2);
          total3 = _mm256_add_epi8(total3, counts3);
        }

        nNeurons -= nVects128 * 128;
        conSyns += nVects128 * (128/4);
        conNeurs += nVects128 * (128/4);
      }
      if(nNeurons >= 64) {
        const __m128i allBitPairs01 = _mm_set1_epi8(0x55);
        const __m256i allBytes3 = _mm256_set1_epi8(0x03);
        const __m128i neurons = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(conNeurs));
        const __m128i synapses = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(conSyns));
        const __m128i vXorred = _mm_xor_si128(neurons, synapses);
        const __m128i synNonzero = _mm_and_si128(synapses, allBitPairs01);
        const __m128i shifted = _mm_slli_epi64(synNonzero, 1);
        const __m128i mask = _mm_or_si128(shifted, allBitPairs01);
        const __m128i increments = _mm_and_si128(vXorred, mask);
        // Now |increments| contain 64 bit pairs, each being -1, 0, or +1

        const __m128i shift2 = _mm_srli_epi64(increments, 2);
        const __m128i shift4 = _mm_srli_epi64(increments, 4);
        const __m128i shift6 = _mm_srli_epi64(increments, 6);

        const __m256i low = _mm256_set_m128i(shift2, increments);
        const __m256i high = _mm256_set_m128i(shift6, shift4);

        const __m256i counts0 = _mm256_and_si256(low, allBytes3);
        const __m256i counts1 = _mm256_and_si256(high, allBytes3);

        total0 = _mm256_add_epi8(total0, counts0);
        total1 = _mm256_add_epi8(total1, counts1);
        nNeurons -= 64;
        conSyns += 64/4;
        conNeurs += 64/4;
      }
      if(nNeurons >= 32) {
        const uint64_t allBitPairs01 = 0x5555555555555555;
        const __m256i allBytes3 = _mm256_set1_epi8(0x03);
        const uint64_t neurons = *reinterpret_cast<const uint64_t*>(conNeurs);
        const uint64_t synapses = *reinterpret_cast<const uint64_t*>(conSyns);
        const uint64_t vXorred = (neurons ^ synapses);
        const uint64_t synNonzero = (synapses & allBitPairs01);
        const uint64_t shifted = (synNonzero << 1);
        const uint64_t mask = (shifted | allBitPairs01);
        const uint64_t increments = (vXorred & mask);
        // Now |increments| contain 32 bit pairs, each being -1, 0, or +1

        const uint64_t shift2 = (increments >> 2);
        const uint64_t shift4 = (increments >> 4);
        const uint64_t shift6 = (increments >> 6);

        const __m256i preCounts = _mm256_setr_epi64x(increments, shift2, shift4, shift6);
        const __m256i counts = _mm256_and_si256(preCounts, allBytes3);

        total2 = _mm256_add_epi8(total2, counts);
        nNeurons -= 32;
        conSyns += 32/4;
        conNeurs += 32/4;
      }
      if(nNeurons >= 16) {
        const uint32_t allBitPairs01 = 0x55555555;
        const __m128i allBytes3 = _mm_set1_epi8(0x03);
        const uint32_t neurons = *reinterpret_cast<const uint32_t*>(conNeurs);
        const uint32_t synapses = *reinterpret_cast<const uint32_t*>(conSyns);
        const uint32_t vXorred = (neurons ^ synapses);
        const uint32_t synNonzero = (synapses & allBitPairs01);
        const uint32_t shifted = (synNonzero << 1);
        const uint32_t mask = (shifted | allBitPairs01);
        const uint32_t increments = (vXorred & mask);
        // Now |increments| contain 16 bit pairs, each being -1, 0, or +1

        const uint32_t shift2 = (increments >> 2);
        const uint32_t shift4 = (increments >> 4);
        const uint32_t shift6 = (increments >> 6);

        const __m128i preCounts = _mm_setr_epi32(increments, shift2, shift4, shift6);
        const __m128i counts128 = _mm_and_si128(preCounts, allBytes3);

        // TODO(srogatch): perhaps a separate counter storing 16 counts would be more efficient, but it's more work
        total3 = _mm256_add_epi8(total3, _mm256_setr_m128i(counts128, _mm_setzero_si128()));
        nNeurons -= 16;
        conSyns += 16/4;
        conNeurs += 16/4;
      }
      if(nNeurons >= 8) {
        const uint16_t allBitPairs01 = 0x5555;
        //const uint64_t allBytes3 = 0x0303030303030303;
        const uint16_t neurons = *reinterpret_cast<const uint16_t*>(conNeurs);
        const uint16_t synapses = *reinterpret_cast<const uint16_t*>(conSyns);
        const uint16_t vXorred = (neurons ^ synapses);
        const uint16_t synNonzero = (synapses & allBitPairs01);
        const uint16_t shifted = (synNonzero << 1);
        const uint16_t mask = (shifted | allBitPairs01);
        const uint16_t increments = (vXorred & mask);
        // Now |increments| contain 8 bit pairs, each being -1, 0, or +1

        modifier += int((increments&3) + ((increments>>2)&3) + ((increments>>4)&3) + ((increments>>6)&3)
          + ((increments>>8)&3) + ((increments>>10)&3) + ((increments>>12)&3) + ((increments>>14)&3));

        nNeurons -= 8;
        conSyns += 8/4;
        conNeurs += 8/4;
      }
      if(nNeurons >= 4) {
        const uint8_t allBitPairs01 = 0x55;
        const uint8_t neurons = *reinterpret_cast<const uint8_t*>(conNeurs);
        const uint8_t synapses = *reinterpret_cast<const uint8_t*>(conSyns);
        const uint8_t vXorred = (neurons ^ synapses);
        const uint8_t synNonzero = (synapses & allBitPairs01);
        const uint8_t shifted = (synNonzero << 1);
        const uint8_t mask = (shifted | allBitPairs01);
        const uint8_t increments = (vXorred & mask);
        // Now |increments| contain 4 bit pairs, each being -1, 0, or +1

        modifier += int((increments&3) + ((increments>>2)&3) + ((increments>>4)&3) + ((increments>>6)&3));

        nNeurons -= 4;
        conSyns += 4/4;
        conNeurs += 4/4;
      }
      assert(nNeurons == 0);
      __m256i totals[4] = {total0, total1, total2, total3};
      __m256i grandTotals[4];
      for(int i=0; i<=3; i++) {
        // Convert the totals from 8- to 32-bit signed integers
        const __m256i vSwapped = _mm256_shuffle_epi32(totals[i], 0b01001110);
        const __m128i xStr0 = _mm256_extracti128_si256(totals[i], 0);
        const __m128i xStr1 = _mm256_extracti128_si256(totals[i], 1);
        const __m128i xSwap0 = _mm256_extracti128_si256(vSwapped, 0);
        const __m128i xSwap1 = _mm256_extracti128_si256(vSwapped, 1);
        const __m256i tot0 = _mm256_cvtepu8_epi32(xStr0);
        const __m256i tot1 = _mm256_cvtepu8_epi32(xStr1);
        const __m256i tot2 = _mm256_cvtepu8_epi32(xSwap0);
        const __m256i tot3 = _mm256_cvtepu8_epi32(xSwap1);
        // Horizontal sum of totals into neurons.input[DATA_LENGTH + inputNeuronIndex]
        const __m256i total01 = _mm256_add_epi32(tot0, tot1);
        const __m256i total23 = _mm256_add_epi32(tot2, tot3);
        grandTotals[i] = _mm256_add_epi32(total01, total23);
      }
      const __m256i gt0 = _mm256_add_epi32(grandTotals[0], grandTotals[1]);
      const __m256i gt1 = _mm256_add_epi32(grandTotals[2], grandTotals[3]);
      const __m256i grandGrandTotal = _mm256_add_epi32(gt0, gt1);
      const __m128i s1 = _mm256_extracti128_si256(grandGrandTotal, 1);
      const __m128i s2 = _mm_add_epi32(s1, _mm256_castsi256_si128(grandGrandTotal));
      (*source_end) += modifier +
        _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
    }

    static void fireNeurons(int* __restrict neuron_ends, const int sourceNeuron, const int nNeurons, const uint8_t* __restrict conSyns,
      uint8_t* __restrict conNeurs)
    {
      assert(nNeurons % 4 == 0);
      const int nBytes = (sourceNeuron / 4);
      const int nPacks =  nBytes * 4;
      // Vectorized prologue
      fireNeuronsVect(neuron_ends+sourceNeuron, conSyns, nPacks, conNeurs);
      updateConNeur(neuron_ends, sourceNeuron, conNeurs);

      // Scalar code handling the total near the source neuron
      {
        assert(nPacks <= sourceNeuron);
        assert(sourceNeuron < nPacks + 4);
        const uint8_t synapses = conSyns[nBytes];
        for (unsigned short anotherNeuronIndex = 0; anotherNeuronIndex < 4; anotherNeuronIndex++)
        {
            int value = neuron_ends[anotherNeuronIndex + nPacks] >= 0 ? 1 : -1;
            value *= ((int(synapses) << (30 - 2*anotherNeuronIndex))>>30);
            neuron_ends[sourceNeuron] += value;
        }
      }
      updateConNeur(neuron_ends, sourceNeuron, conNeurs);

      // Vector epilogue
      fireNeuronsVect(neuron_ends+sourceNeuron, conSyns+nBytes+1, nNeurons-nPacks-4, conNeurs+nBytes+1);
      updateConNeur(neuron_ends, sourceNeuron, conNeurs);
    }

    static void setZeroConsyn(uint8_t* buffer, const int index) {
      const int iChar = (index >> 2);
      const int iPack = (index & 3);
      buffer[iChar] &= ~(3<<(iPack<<1));
    }

    template<int hint> static void prefetch(const void* start, const int size) {
      const uint8_t* p = static_cast<const uint8_t*>(start);
      const uint8_t* end = p+size;
      while(p < end) {
        _mm_prefetch(p, hint);
        p += 64;
      }
    }

    // Produce 01 bit pairs for negatives and 11 for positives or zero
    static void condenseNeurons(const int* __restrict input, const int nNeurons, uint8_t* __restrict output) {
      constexpr const int nNeuronsPerVector = gcVecBytes / sizeof(input[0]);
      // Let it a bit overflow
      const int nVect = (nNeurons + nNeuronsPerVector - 1) / nNeuronsPerVector;
      int i=0;
      for(; i<nVect; i++) {
        const __m256i inpNeurs = _mm256_load_si256(reinterpret_cast<const __m256i*>(input)+i);
        const int negatives = _mm256_movemask_ps(_mm256_castsi256_ps(inpNeurs));
        // This produces 10 for negatives and 00 for positives
        const uint16_t inverse = _pdep_u32(negatives, 0xAAAA);
        // Change it to 01 for negatives and 11 for positives
        const uint16_t i2bitsPerNeurons = inverse ^ 0xffff;
        *(reinterpret_cast<uint16_t*>(output)+i) = i2bitsPerNeurons;
      }
    }

    static void updateConNeur(const int *neuron_ends, const int sourceNeuron, uint8_t* conNeurs) {
      const uint8_t bitPair = 3 ^ ((neuron_ends[sourceNeuron] >> 30) & 2);
      const int iByte = (sourceNeuron >> 2);
      const int iPack = (sourceNeuron & 3);
      conNeurs[iByte] = (conNeurs[iByte] & ~(3<<(iPack<<1))) | (bitPair<<(iPack<<1));
    }

    bool findSolution(unsigned char nonce[32])
    {
        alignas(64) struct
        {
            int input[DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH];
            int output[INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH];
        } neurons;
        alignas(64) struct
        {
            char input[(NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH)];
            char output[(NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH)];
            unsigned short lengths[MAX_INPUT_DURATION * (NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + MAX_OUTPUT_DURATION * (NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH)];
        } synapses;
        alignas(64) struct
        {
          uint8_t input[(NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) / 4];
          uint8_t output[(NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) / 4];
        } conSyns;
        alignas(64) struct
        {
          uint8_t input[(DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) / 4];
          uint8_t output[(INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) / 4];
        } conNeurs;


        _rdrand64_step((unsigned long long*) & nonce[0]);
        _rdrand64_step((unsigned long long*) & nonce[8]);
        _rdrand64_step((unsigned long long*) & nonce[16]);
        _rdrand64_step((unsigned long long*) & nonce[24]);
        random(computorPublicKey, nonce, (unsigned char*)&synapses, sizeof(synapses));

        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; inputNeuronIndex++)
        {
            // for (unsigned int anotherInputNeuronIndex = 0; anotherInputNeuronIndex < DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; anotherInputNeuronIndex++)
            // {
            //     const unsigned int offset = inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + anotherInputNeuronIndex;
            //     synapses.input[offset] = (((unsigned char)synapses.input[offset]) % 3) - 1;
            // }
            static_assert((DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) % 4 == 0, "Must be a multiple of 4");
            populateSynapses(synapses.input+inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH),
              DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH,
              conSyns.input + inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) / 4);
        }
        for (unsigned int outputNeuronIndex = 0; outputNeuronIndex < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; outputNeuronIndex++)
        {
            // for (unsigned int anotherOutputNeuronIndex = 0; anotherOutputNeuronIndex < INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; anotherOutputNeuronIndex++)
            // {
            //     const unsigned int offset = outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + anotherOutputNeuronIndex;
            //     synapses.output[offset] = (((unsigned char)synapses.output[offset]) % 3) - 1;
            // }
            static_assert((INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) % 4 == 0, "Must be a multiple of 4");
            populateSynapses(synapses.output + outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH),
              INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH,
              conSyns.output + outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) / 4);
        }
        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; inputNeuronIndex++)
        {
            // synapses.input[inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + (DATA_LENGTH + inputNeuronIndex)] = 0;
            setZeroConsyn(conSyns.input, inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + (DATA_LENGTH + inputNeuronIndex));
        }
        for (unsigned int outputNeuronIndex = 0; outputNeuronIndex < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; outputNeuronIndex++)
        {
            //synapses.output[outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + (INFO_LENGTH + outputNeuronIndex)] = 0;
            setZeroConsyn(conSyns.output, outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + (INFO_LENGTH + outputNeuronIndex));
        }

        unsigned int lengthIndex = 0;

        prefetch<_MM_HINT_T0>(neurons.input, sizeof(neurons.input));
        prefetch<_MM_HINT_T1>(conSyns.input, sizeof(conSyns.input));
        memcpy(&neurons.input[0], &data, sizeof(data));
        memset(&neurons.input[sizeof(data) / sizeof(neurons.input[0])], 0, sizeof(neurons) - sizeof(data));
        condenseNeurons(neurons.input, sizeof(neurons.input)/sizeof(neurons.input[0]), conNeurs.input);
        prefetch<_MM_HINT_T0>(conNeurs.input, sizeof(conNeurs.input));
        for (unsigned int tick = 0; tick < MAX_INPUT_DURATION; tick++)
        {
            unsigned short neuronIndices[NUMBER_OF_INPUT_NEURONS + INFO_LENGTH];
            unsigned short numberOfRemainingNeurons = 0;
            for (numberOfRemainingNeurons = 0; numberOfRemainingNeurons < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; numberOfRemainingNeurons++)
            {
                neuronIndices[numberOfRemainingNeurons] = numberOfRemainingNeurons;
            }
            while (numberOfRemainingNeurons)
            {
                const unsigned short neuronIndexIndex = synapses.lengths[lengthIndex++] % numberOfRemainingNeurons;
                const unsigned short inputNeuronIndex = neuronIndices[neuronIndexIndex];
                neuronIndices[neuronIndexIndex] = neuronIndices[--numberOfRemainingNeurons];
                // for (unsigned short anotherInputNeuronIndex = 0; anotherInputNeuronIndex < DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; anotherInputNeuronIndex++)
                // {
                //     int value = neurons.input[anotherInputNeuronIndex] >= 0 ? 1 : -1;
                //     value *= synapses.input[inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + anotherInputNeuronIndex];
                //     neurons.input[DATA_LENGTH + inputNeuronIndex] += value;
                // }
                auto pConSyns = conSyns.input + inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) / 4;
                prefetch<_MM_HINT_T0>(pConSyns, (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH)/4);
                fireNeurons(neurons.input, DATA_LENGTH+inputNeuronIndex, DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH,
                  pConSyns, conNeurs.input);
            }
        }

        prefetch<_MM_HINT_T0>(neurons.output, sizeof(neurons.output));
        prefetch<_MM_HINT_T1>(conSyns.output, sizeof(conSyns.output));
        memcpy(&neurons.output[0], &neurons.input[DATA_LENGTH + NUMBER_OF_INPUT_NEURONS], INFO_LENGTH * sizeof(neurons.input[0]));
        condenseNeurons(neurons.output, sizeof(neurons.output)/sizeof(neurons.output[0]), conNeurs.output);
        prefetch<_MM_HINT_T0>(conNeurs.output, sizeof(conNeurs.output));
        for (unsigned int tick = 0; tick < MAX_OUTPUT_DURATION; tick++)
        {
            unsigned short neuronIndices[NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH];
            unsigned short numberOfRemainingNeurons = 0;
            for (numberOfRemainingNeurons = 0; numberOfRemainingNeurons < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; numberOfRemainingNeurons++)
            {
                neuronIndices[numberOfRemainingNeurons] = numberOfRemainingNeurons;
            }
            while (numberOfRemainingNeurons)
            {
                const unsigned short neuronIndexIndex = synapses.lengths[lengthIndex++] % numberOfRemainingNeurons;
                const unsigned short outputNeuronIndex = neuronIndices[neuronIndexIndex];
                neuronIndices[neuronIndexIndex] = neuronIndices[--numberOfRemainingNeurons];
                // for (unsigned int anotherOutputNeuronIndex = 0; anotherOutputNeuronIndex < INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; anotherOutputNeuronIndex++)
                // {
                //     int value = neurons.output[anotherOutputNeuronIndex] >= 0 ? 1 : -1;
                //     value *= synapses.output[outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + anotherOutputNeuronIndex];
                //     neurons.output[INFO_LENGTH + outputNeuronIndex] += value;
                // }
                auto pConSyns = conSyns.output + outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) / 4;
                prefetch<_MM_HINT_T0>(pConSyns, (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH)/4);
                fireNeurons(neurons.output, INFO_LENGTH+outputNeuronIndex, INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH,
                  pConSyns, conNeurs.output);
            }
        }

        unsigned int score = 0;

        if(gVerifySolution) {
          verifySolution(nonce, neurons.output, conSyns.input, conSyns.output);
        }

        for (unsigned int i = 0; i < DATA_LENGTH; i++)
        {
            if ((data[i] >= 0) == (neurons.output[INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + i] >= 0))
            {
                score++;
            }
        }

        return score >= SOLUTION_THRESHOLD;
    }

    static char decodeConsyn(uint8_t* buffer, const int index) {
      const int iChar = (index >> 2);
      const int iPack = (index & 3);
      char ans = (int(buffer[iChar]) << (30 - 2*iPack)) >> 30;
      return ans;
    }

    void __attribute__ ((noinline)) verifySolution(unsigned char nonce[32], int* neuronOutputs, uint8_t* csInput, uint8_t* csOutput) {
        uint64_t testConSyns =  0b0000111100000101;
        uint64_t testConNeurs = 0b1111011101110111;
        int accum = 0;
        fireNeuronsVect(&accum, (uint8_t*)&testConSyns, 8, (uint8_t*)&testConNeurs);
        assert(accum == 0);
        struct
        {
            int input[DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH];
            int output[INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH];
        } neurons;
        struct
        {
            char input[(NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH)];
            char output[(NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH)];
            unsigned short lengths[MAX_INPUT_DURATION * (NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + MAX_OUTPUT_DURATION * (NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH)];
        } synapses;

        // _rdrand64_step((unsigned long long*) & nonce[0]);
        // _rdrand64_step((unsigned long long*) & nonce[8]);
        // _rdrand64_step((unsigned long long*) & nonce[16]);
        // _rdrand64_step((unsigned long long*) & nonce[24]);
        random(computorPublicKey, nonce, (unsigned char*)&synapses, sizeof(synapses));
        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; inputNeuronIndex++)
        {
            for (unsigned int anotherInputNeuronIndex = 0; anotherInputNeuronIndex < DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; anotherInputNeuronIndex++)
            {
                const unsigned int offset = inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + anotherInputNeuronIndex;
                synapses.input[offset] = (((unsigned char)synapses.input[offset]) % 3) - 1;
            }
        }
        for (unsigned int outputNeuronIndex = 0; outputNeuronIndex < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; outputNeuronIndex++)
        {
            for (unsigned int anotherOutputNeuronIndex = 0; anotherOutputNeuronIndex < INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; anotherOutputNeuronIndex++)
            {
                const unsigned int offset = outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + anotherOutputNeuronIndex;
                synapses.output[offset] = (((unsigned char)synapses.output[offset]) % 3) - 1;
            }
        }
        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; inputNeuronIndex++)
        {
            synapses.input[inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + (DATA_LENGTH + inputNeuronIndex)] = 0;
        }
        for (unsigned int outputNeuronIndex = 0; outputNeuronIndex < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; outputNeuronIndex++)
        {
            synapses.output[outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + (INFO_LENGTH + outputNeuronIndex)] = 0;
        }

        for (unsigned int inputNeuronIndex = 0; inputNeuronIndex < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; inputNeuronIndex++)
        {
            for (unsigned int anotherInputNeuronIndex = 0; anotherInputNeuronIndex < DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; anotherInputNeuronIndex++)
            {
                const unsigned int offset = inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + anotherInputNeuronIndex;
                char decoded = decodeConsyn(csInput, offset);
                if(synapses.input[offset] != decoded) {
                  printf(" SI:%d,A:%d,E:%d ", int(offset), int(decoded), int(synapses.input[offset]));
                }
            }
        }
        for (unsigned int outputNeuronIndex = 0; outputNeuronIndex < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; outputNeuronIndex++)
        {
            for (unsigned int anotherOutputNeuronIndex = 0; anotherOutputNeuronIndex < INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; anotherOutputNeuronIndex++)
            {
                const unsigned int offset = outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + anotherOutputNeuronIndex;
                char decoded = decodeConsyn(csOutput, offset);
                if(synapses.output[offset] != decoded) {
                  printf(" SO:%d,A:%d,E:%d ", int(offset), int(decoded), int(synapses.output[offset]));
                }
            }
        }

        unsigned int lengthIndex = 0;

        memcpy(&neurons.input[0], &data, sizeof(data));
        memset(&neurons.input[sizeof(data) / sizeof(neurons.input[0])], 0, sizeof(neurons) - sizeof(data));

        for (unsigned int tick = 0; tick < MAX_INPUT_DURATION; tick++)
        {
            unsigned short neuronIndices[NUMBER_OF_INPUT_NEURONS + INFO_LENGTH];
            unsigned short numberOfRemainingNeurons = 0;
            for (numberOfRemainingNeurons = 0; numberOfRemainingNeurons < NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; numberOfRemainingNeurons++)
            {
                neuronIndices[numberOfRemainingNeurons] = numberOfRemainingNeurons;
            }
            while (numberOfRemainingNeurons)
            {
                const unsigned short neuronIndexIndex = synapses.lengths[lengthIndex++] % numberOfRemainingNeurons;
                const unsigned short inputNeuronIndex = neuronIndices[neuronIndexIndex];
                neuronIndices[neuronIndexIndex] = neuronIndices[--numberOfRemainingNeurons];
                for (unsigned short anotherInputNeuronIndex = 0; anotherInputNeuronIndex < DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH; anotherInputNeuronIndex++)
                {
                    int value = neurons.input[anotherInputNeuronIndex] >= 0 ? 1 : -1;
                    value *= synapses.input[inputNeuronIndex * (DATA_LENGTH + NUMBER_OF_INPUT_NEURONS + INFO_LENGTH) + anotherInputNeuronIndex];
                    neurons.input[DATA_LENGTH + inputNeuronIndex] += value;
                }
            }
        }

        memcpy(&neurons.output[0], &neurons.input[DATA_LENGTH + NUMBER_OF_INPUT_NEURONS], INFO_LENGTH * sizeof(neurons.input[0]));

        for (unsigned int tick = 0; tick < MAX_OUTPUT_DURATION; tick++)
        {
            unsigned short neuronIndices[NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH];
            unsigned short numberOfRemainingNeurons = 0;
            for (numberOfRemainingNeurons = 0; numberOfRemainingNeurons < NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; numberOfRemainingNeurons++)
            {
                neuronIndices[numberOfRemainingNeurons] = numberOfRemainingNeurons;
            }
            while (numberOfRemainingNeurons)
            {
                const unsigned short neuronIndexIndex = synapses.lengths[lengthIndex++] % numberOfRemainingNeurons;
                const unsigned short outputNeuronIndex = neuronIndices[neuronIndexIndex];
                neuronIndices[neuronIndexIndex] = neuronIndices[--numberOfRemainingNeurons];
                for (unsigned int anotherOutputNeuronIndex = 0; anotherOutputNeuronIndex < INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH; anotherOutputNeuronIndex++)
                {
                    int value = neurons.output[anotherOutputNeuronIndex] >= 0 ? 1 : -1;
                    value *= synapses.output[outputNeuronIndex * (INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + DATA_LENGTH) + anotherOutputNeuronIndex];
                    neurons.output[INFO_LENGTH + outputNeuronIndex] += value;
                }
            }
        }

        for (unsigned int i = 0; i < DATA_LENGTH; i++)
        {
            const int iNeuron = INFO_LENGTH + NUMBER_OF_OUTPUT_NEURONS + i;
            if (neurons.output[iNeuron] != neuronOutputs[iNeuron])
            {
              printf(" F:%d,A:%d,E:%d ", iNeuron, neuronOutputs[iNeuron], neurons.output[iNeuron]);
            }
        }
    }
};

static volatile char state = 0;

static unsigned char computorPublicKey[32] __attribute__((aligned(32))) = {
  151,146,231,244,235,47,109,60,237,8,73,9,158,250,191,113,120,148,238,230,209,48,186,139,48,72,99,39,221,137,252,47};
static unsigned char nonce[32] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static volatile long long numberOfMiningIterations = 0;
static unsigned int numberOfFoundSolutions = 0;

static void ctrlCHandlerRoutine([[maybe_unused]] int sig)
{
    state = 1;
}

void mySleep(int sleepMs)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
}

uint64_t getTimeMs(void)
{
    struct timeval tv;

    gettimeofday(&tv, 0);
    return uint64_t(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

static uint64_t GetTickCountMs()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t)(ts.tv_nsec / 1000000) + ((uint64_t)ts.tv_sec * 1000ull);
}

static void *miningThreadProc([[maybe_unused]] void *ptr)
{
    Miner miner;
    miner.initialize();
    miner.setComputorPublicKey(computorPublicKey);

    alignas(32) unsigned char nonce[32];
    while (!state)
    {
        if (miner.findSolution(nonce))
        {
            while (!EQUAL(*((__m256i*)::nonce), ZERO))
            {
                mySleep(1);
            }
            *((__m256i*)::nonce) = *((__m256i*)nonce);
            numberOfFoundSolutions++;
        }

        __sync_fetch_and_add(&numberOfMiningIterations, 1);
    }

    return NULL;
}

static bool sendData(int serverSocket, char *buffer, unsigned int size)
{
    while (size)
    {
        int numberOfBytes;
        if ((numberOfBytes = send(serverSocket, buffer, size, 0)) <= 0)
        {
            strerror(errno);
            return false;
        }
        buffer += numberOfBytes;
        size -= numberOfBytes;
    }

    return true;
}

static bool getPublicKeyFromIdentity(const unsigned char *id, unsigned char *publicKey)
{
    unsigned char publicKeyBuffer[32];
    for (int i = 0; i < 4; i++)
    {
        *((unsigned long long *)&publicKeyBuffer[i << 3]) = 0;
        for (int j = 14; j-- > 0;)
        {
            if (id[i * 14 + j] < 'A' || id[i * 14 + j] > 'Z')
            {
                return false;
            }

            *((unsigned long long *)&publicKeyBuffer[i << 3]) = *((unsigned long long *)&publicKeyBuffer[i << 3]) * 26 + (id[i * 14 + j] - 'A');
        }
    }
    *((__m256i *)publicKey) = *((__m256i *)publicKeyBuffer);

    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage:   Qiner <IP-address> <Id> <Number of threads> [verify]\n");
    }
    else
    {
        printf("Qiner %d.2 is launched.\n", EPOCH);

        signal(SIGINT, ctrlCHandlerRoutine);

        if (!getPublicKeyFromIdentity((const unsigned char *)argv[2], computorPublicKey))
        {
            printf("The Id is invalid!\n");
        }
        else
        {
            if(argc >= 5) {
              if(std::string(argv[4]) == "verify") {
                gVerifySolution = true;
              }
            }

            unsigned int numberOfThreads = 1;
            numberOfThreads = atoi(argv[3]);

            printf("%d threads are used.\n", numberOfThreads);
            pthread_t *my_thread = new pthread_t[numberOfThreads];

            for (unsigned int i = 0; i < numberOfThreads; ++i) {
                pthread_attr_t attr;
                size_t stackSize = 25165824 * (1.25 + (gVerifySolution ? 1.0 : 0.0));

                if (pthread_attr_init(&attr) != 0) {
                    perror("Failed to initialise pthread attributes\n");
                    exit(EXIT_FAILURE);
                }

                if (pthread_attr_setstacksize(&attr, stackSize) != 0) {
                    perror("Failed to set stack size\n");
                    exit(EXIT_FAILURE);
                }

                if (pthread_create(&my_thread[i], &attr, miningThreadProc, NULL) != 0) {
                    perror("Failed to create thread\n");
                    exit(EXIT_FAILURE);
                }

                if (pthread_attr_destroy(&attr) != 0) {
                    perror("Failed to destroy the pthread attribute object\n");
                    exit(EXIT_FAILURE);
                }
            }

            // When threads are launched, no need to keep pointer to heap
            delete[] my_thread;

            unsigned long long timestamp = GetTickCountMs();

            long long prevNumberOfMiningIterations = 0;
            while (!state)
            {
                if (!EQUAL(*((__m256i *)nonce), ZERO))
                {
                    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
                    struct timeval tv;
                    tv.tv_sec = 5;
                    tv.tv_usec = 0;
                    setsockopt(serverSocket, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof tv);
                    if (serverSocket < 0)
                    {
                        printf("Failed to create a socket!\n");
                    }
                    else

                    {
                        sockaddr_in addr;

                        explicit_bzero((char *)&addr, sizeof(addr));
                        addr.sin_family = AF_INET;
                        addr.sin_port = htons(PORT);

                        if (inet_pton(AF_INET, argv[1], &addr.sin_addr) <= 0)
                        {
                            printf("Error translating command line ip address to usable one.");
                        }

                        if (connect(serverSocket, (const sockaddr *)&addr, sizeof(addr)) < 0)
                        {
                            printf("Failed to connect, done here.\n");
                        }
                        else
                        {
                            struct
                            {
                                RequestResponseHeader header;
                                Message message;
                                unsigned char solutionNonce[32];
                                unsigned char signature[64];
                            } packet;

                            packet.header.setSize(sizeof(packet));
                            packet.header.setProtocol();
                            packet.header.zeroDejavu();
                            packet.header.setType(BROADCAST_MESSAGE);

                            *((__m256i*)packet.message.sourcePublicKey) = ZERO;
                            *((__m256i*)packet.message.destinationPublicKey) = *((__m256i*)computorPublicKey);

                            unsigned char sharedKeyAndGammingNonce[64];
                            explicit_bzero((char *)&sharedKeyAndGammingNonce, 32);
                            unsigned char gammingKey[32];
                            do
                            {
                                _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[0]);
                                _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[8]);
                                _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[16]);
                                _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[24]);
                                memcpy(&sharedKeyAndGammingNonce[32], packet.message.gammingNonce, 32);
                                KangarooTwelve64To32(sharedKeyAndGammingNonce, gammingKey);
                            } while (gammingKey[0]);

                            unsigned char gamma[32];
                            KangarooTwelve(gammingKey, sizeof(gammingKey), gamma, sizeof(gamma));
                            for (unsigned int i = 0; i < 32; i++)
                            {
                                packet.solutionNonce[i] = nonce[i] ^ gamma[i];
                            }

                            _rdrand64_step((unsigned long long*)&packet.signature[0]);
                            _rdrand64_step((unsigned long long*)&packet.signature[8]);
                            _rdrand64_step((unsigned long long*)&packet.signature[16]);
                            _rdrand64_step((unsigned long long*)&packet.signature[24]);
                            _rdrand64_step((unsigned long long*)&packet.signature[32]);
                            _rdrand64_step((unsigned long long*)&packet.signature[40]);
                            _rdrand64_step((unsigned long long*)&packet.signature[48]);
                            _rdrand64_step((unsigned long long*)&packet.signature[56]);
                            if (sendData(serverSocket, (char*)&packet, packet.header.size()))
                            {
                                for (unsigned int i = 0; i > 10; i++)
                                {
                                    sendData(serverSocket, (char*)&packet, packet.header.size());
                                    mySleep(500);
                                }
                                *((__m256i*)nonce) = ZERO;
                            }
                        }
                        close(serverSocket);
                    }
                }

                constexpr const int repPeriodMs = 10000;
                mySleep(repPeriodMs);

                unsigned long long delta = GetTickCountMs() - timestamp;
                if (delta >= repPeriodMs)
                {
                    time_t rawTime = time(0);
                    tm *systemTime = localtime(&rawTime);

                    printf("|   %d-%d%d-%d%d %d%d:%d%d:%d%d   |   %lli it/s   |   %i solutions   |   %.6s...%s   |\n",
                      systemTime->tm_year + 1900, ((int)systemTime->tm_mon + 1) / 10, ((int)systemTime->tm_mon + 1) % 10,
                      (int)systemTime->tm_mday / 10, (int)systemTime->tm_mday % 10, (int)systemTime->tm_hour / 10,
                      (int)systemTime->tm_hour % 10, (int)systemTime->tm_min / 10, (int)systemTime->tm_min % 10,
                      (int)systemTime->tm_sec / 10, (int)systemTime->tm_sec % 10,
                      (numberOfMiningIterations - prevNumberOfMiningIterations) * 1000 / delta,
                      numberOfFoundSolutions, argv[2], argv[2] + strlen((const char*)argv[2]) - 6);

                    prevNumberOfMiningIterations = numberOfMiningIterations;
                    timestamp = GetTickCountMs();
                }
            }
        }

        printf("Qiner %d.2 is shut down.\n", EPOCH);
    }
    return 0;
}