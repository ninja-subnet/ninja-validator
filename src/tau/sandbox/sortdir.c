/* libsortdir.so -- LD_PRELOAD shim for deterministic directory reads.
 *
 * Returns directory entries in byte-sorted (strcmp) order so that find / ls /
 * shell globs / Python os.listdir observe a stable order regardless of the
 * underlying filesystem's readdir order. This removes the dominant source of
 * run-to-run "observation" divergence in the solver container, which otherwise
 * defeats response-cache replay.
 *
 * Targets glibc on linux/amd64 (LP64: struct dirent and struct dirent64 share
 * layout, so a single sorted buffer serves both readdir and readdir64).
 *
 * Strategy: on the first readdir for a DIR*, drain the whole directory via the
 * real readdir64, keep byte-sorted copies, and hand them back one at a time.
 * Falls back to the real function on any allocation failure or table overflow.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STREAMS 8192

typedef struct {
    DIR *dir;
    struct dirent **ents;
    size_t count;
    size_t pos;
    int loaded;
} stream_t;

static stream_t g_streams[MAX_STREAMS];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static struct dirent *(*real_readdir)(DIR *) = NULL;
static struct dirent64 *(*real_readdir64)(DIR *) = NULL;
static int (*real_closedir)(DIR *) = NULL;

static void ensure_syms(void) {
    if (!real_readdir) real_readdir = (struct dirent *(*)(DIR *))dlsym(RTLD_NEXT, "readdir");
    if (!real_readdir64) real_readdir64 = (struct dirent64 *(*)(DIR *))dlsym(RTLD_NEXT, "readdir64");
    if (!real_closedir) real_closedir = (int (*)(DIR *))dlsym(RTLD_NEXT, "closedir");
}

static int ent_cmp(const void *a, const void *b) {
    const struct dirent *x = *(const struct dirent *const *)a;
    const struct dirent *y = *(const struct dirent *const *)b;
    return strcmp(x->d_name, y->d_name);
}

static stream_t *slot_for(DIR *d, int create) {
    size_t i;
    for (i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].dir == d) return &g_streams[i];
    if (!create) return NULL;
    for (i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].dir == NULL) {
            g_streams[i].dir = d;
            g_streams[i].ents = NULL;
            g_streams[i].count = 0;
            g_streams[i].pos = 0;
            g_streams[i].loaded = 0;
            return &g_streams[i];
        }
    return NULL;
}

/* Drain via the real readdir64 and store byte-sorted copies. 0 on success. */
static int load_stream(stream_t *s, DIR *d) {
    size_t cap = 64;
    struct dirent64 *e;
    s->ents = (struct dirent **)malloc(cap * sizeof(*s->ents));
    if (!s->ents) return -1;
    s->count = 0;
    while ((e = real_readdir64(d)) != NULL) {
        struct dirent64 *copy;
        if (s->count == cap) {
            struct dirent **n;
            cap *= 2;
            n = (struct dirent **)realloc(s->ents, cap * sizeof(*s->ents));
            if (!n) return -1;
            s->ents = n;
        }
        copy = (struct dirent64 *)malloc(e->d_reclen);
        if (!copy) return -1;
        memcpy(copy, e, e->d_reclen);
        s->ents[s->count++] = (struct dirent *)copy;
    }
    qsort(s->ents, s->count, sizeof(*s->ents), ent_cmp);
    s->loaded = 1;
    s->pos = 0;
    return 0;
}

struct dirent *readdir(DIR *d) {
    struct dirent *ret;
    stream_t *s;
    ensure_syms();
    if (!real_readdir64) return real_readdir ? real_readdir(d) : NULL;
    pthread_mutex_lock(&g_lock);
    s = slot_for(d, 1);
    if (!s) {
        pthread_mutex_unlock(&g_lock);
        return real_readdir ? real_readdir(d) : NULL;
    }
    if (!s->loaded && load_stream(s, d) != 0) {
        s->loaded = 1;
        s->pos = s->count;
    }
    ret = (s->pos < s->count) ? s->ents[s->pos++] : NULL;
    pthread_mutex_unlock(&g_lock);
    return ret;
}

struct dirent64 *readdir64(DIR *d) {
    struct dirent64 *ret;
    stream_t *s;
    ensure_syms();
    if (!real_readdir64) return NULL;
    pthread_mutex_lock(&g_lock);
    s = slot_for(d, 1);
    if (!s) {
        pthread_mutex_unlock(&g_lock);
        return real_readdir64(d);
    }
    if (!s->loaded && load_stream(s, d) != 0) {
        s->loaded = 1;
        s->pos = s->count;
    }
    ret = (s->pos < s->count) ? (struct dirent64 *)s->ents[s->pos++] : NULL;
    pthread_mutex_unlock(&g_lock);
    return ret;
}

int closedir(DIR *d) {
    stream_t *s;
    ensure_syms();
    pthread_mutex_lock(&g_lock);
    s = slot_for(d, 0);
    if (s) {
        size_t i;
        for (i = 0; i < s->count; i++) free(s->ents[i]);
        free(s->ents);
        s->dir = NULL;
        s->ents = NULL;
        s->count = 0;
        s->pos = 0;
        s->loaded = 0;
    }
    pthread_mutex_unlock(&g_lock);
    return real_closedir ? real_closedir(d) : 0;
}
