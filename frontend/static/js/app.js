/* ================================================================
   Custom Resume — Vue 3 SPA (CDN Build)
   HELM-inspired dark command-center UI
   ================================================================ */

const { createApp, ref, reactive, computed, watch, onMounted, onUnmounted, nextTick, h, provide, inject } = Vue;
const { createRouter, createWebHashHistory, useRouter, useRoute, onBeforeRouteLeave } = VueRouter;
const { createPinia, defineStore } = Pinia;

// ================================================================
// API CLIENT — with automatic 401 refresh+retry
// ================================================================
const api = {
  _base: '',
  _refreshing: null, // shared promise so concurrent 401s don't spam refresh

  _getHeaders() {
    const headers = { 'Accept': 'application/json' };
    const token = localStorage.getItem('token');
    if (token) headers['Authorization'] = `Bearer ${token}`;
    return headers;
  },

  // Core request method — all verbs go through here
  async _request(url, opts = {}, _isRetry = false) {
    const res = await fetch(this._base + url, opts);

    if (res.status === 401 && !_isRetry && url !== '/auth/refresh') {
      const body = await res.json().catch(() => ({}));
      // Only attempt refresh if the token expired (not if it's invalid/missing)
      if (body.detail === 'token_expired') {
        const refreshed = await this._tryRefresh();
        if (refreshed) {
          // Rebuild auth header with new token and retry once
          const newToken = localStorage.getItem('token');
          if (opts.headers && newToken) opts.headers['Authorization'] = `Bearer ${newToken}`;
          return this._request(url, opts, true);
        }
      }
      // Refresh failed or token is invalid — kick to login
      this._forceLogout();
      throw new Error(body.detail || 'Session expired');
    }

    if (!res.ok) throw await this._parseError(res);
    if (res.status === 204) return null;
    return res.json();
  },

  async _tryRefresh() {
    // Deduplicate: if a refresh is already in-flight, all callers share the same promise
    if (this._refreshing) return this._refreshing;

    this._refreshing = (async () => {
      try {
        const headers = { ...this._getHeaders(), 'Content-Type': 'application/json' };
        const res = await fetch(this._base + '/auth/refresh', { method: 'POST', headers });
        if (!res.ok) return false;
        const data = await res.json();
        if (data.access_token) {
          localStorage.setItem('token', data.access_token);
          return true;
        }
        return false;
      } catch {
        return false;
      } finally {
        this._refreshing = null;
      }
    })();

    return this._refreshing;
  },

  _forceLogout() {
    localStorage.removeItem('token');
    // Don't set window.location.hash here — it races with the router guard.
    // The guard will redirect to /login on the next navigation when it finds
    // no token and no auth.user.
  },

  async _parseError(res) {
    let msg;
    try {
      const data = await res.json();
      msg = data.detail || `HTTP ${res.status}`;
    } catch {
      msg = `HTTP ${res.status}`;
    }
    const err = new Error(msg);
    err.status = res.status;
    return err;
  },

  async get(url) {
    return this._request(url, { headers: this._getHeaders() });
  },
  async post(url, body) {
    const headers = { ...this._getHeaders(), 'Content-Type': 'application/json' };
    return this._request(url, { method: 'POST', headers, body: body !== undefined ? JSON.stringify(body) : undefined });
  },
  async put(url, body) {
    const headers = { ...this._getHeaders(), 'Content-Type': 'application/json' };
    return this._request(url, { method: 'PUT', headers, body: JSON.stringify(body) });
  },
  async del(url) {
    return this._request(url, { method: 'DELETE', headers: this._getHeaders() });
  },
  async upload(url, file) {
    const form = new FormData();
    form.append('file', file);
    const headers = {};
    const token = localStorage.getItem('token');
    if (token) headers['Authorization'] = `Bearer ${token}`;
    return this._request(url, { method: 'POST', headers, body: form });
  },
};

// ================================================================
// SHARED UTILITIES
// ================================================================
function buildPdfFilename(candidateName, role, company, jobId) {
  if (candidateName && role && company) return `${candidateName} - ${role} (${company}).pdf`;
  if (role && company) return `${role} (${company}).pdf`;
  return `resume_${jobId}.pdf`;
}

function jobLabel(j) {
  const jd = j.job_description;
  if (jd && jd.role && jd.company) return `${jd.role} at ${jd.company}`;
  if (jd && jd.role) return jd.role;
  return 'Resume Job';
}

/** Format an ISO date string (UTC from Postgres) to the client's local timezone. */
function formatDate(iso, { includeYear = false, includeTime = false } = {}) {
  if (!iso) return '-';
  // Postgres returns timezone-aware strings like "2026-02-16T10:30:00+00:00"
  // or naive strings like "2026-02-16T10:30:00.000000".
  // If no timezone info at all, append Z to interpret as UTC.
  let dateStr = iso;
  if (!/[Z+\-]\d{2}:\d{2}$/.test(dateStr) && !dateStr.endsWith('Z')) {
    dateStr += 'Z';
  }
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return '-';
  const opts = { month: 'short', day: 'numeric' };
  if (includeYear) opts.year = 'numeric';
  if (includeTime) { opts.hour = 'numeric'; opts.minute = '2-digit'; }
  return d.toLocaleString(undefined, opts);
}

async function downloadJobPdf(jobId, candidateName, role, company) {
  const token = localStorage.getItem('token');
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`/jobs/${jobId}/pdf`, { headers });
  if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = buildPdfFilename(candidateName, role, company, jobId).replace(/[/\\?%*:|"<>]/g, '');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ================================================================
// STORES
// ================================================================
// Keep fetch promises outside reactive state to avoid Vue proxy issues
let _profileFetchPromise = null;
let _jobFetchPromise = null;

const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    loading: false,
    error: null,
  }),
  getters: {
    isLoggedIn: (s) => !!s.user,
    isSuperAdmin: (s) => !!(s.user && s.user.is_super_admin),
    tenantName: (s) => (s.user && s.user.tenant_name) || null,
    initials: (s) => {
      if (!s.user || !s.user.name) return '?';
      return s.user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    },
  },
  actions: {
    async fetchMe() {
      this.loading = true;
      this.error = null;
      try {
        this.user = await api.get('/auth/me');
      } catch (e) {
        this.user = null;
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
    logout() {
      this.user = null;
      localStorage.removeItem('token');
    },
  },
});

const useProfileStore = defineStore('profile', {
  state: () => ({
    profiles: [],
    current: null,
    selectedId: null,
    loading: false,
    uploading: false,
    error: null,
    page: 1,
    totalPages: 1,
    total: 0,
    limit: 10,
  }),
  actions: {
    selectProfile(id) {
      this.selectedId = id;
    },
    async fetchProfiles(force = false, { page, limit } = {}) {
      if (_profileFetchPromise) return _profileFetchPromise;
      if (!force && this.profiles.length > 0) return;
      const p = page || this.page;
      const l = limit || this.limit;
      this.loading = true;
      _profileFetchPromise = api.get(`/profiles/?page=${p}&limit=${l}`)
        .then(data => {
          this.profiles = data.items;
          this.total = data.total;
          this.page = data.page;
          this.totalPages = data.pages;
          this.limit = data.limit;
        })
        .catch(e => { this.error = e.message; })
        .finally(() => { this.loading = false; _profileFetchPromise = null; });
      return _profileFetchPromise;
    },
    async goToPage(n) {
      this.page = n;
      await this.fetchProfiles(true, { page: n });
    },
    async fetchProfile(id) {
      this.loading = true;
      this.current = null;
      try {
        this.current = await api.get(`/profiles/${id}`);
      } catch (e) { this.error = e.message; }
      finally { this.loading = false; }
    },
    async uploadProfile(file) {
      this.uploading = true;
      this.error = null;
      try {
        const result = await api.upload('/profiles/upload', file);
        this.page = 1;
        await this.fetchProfiles(true);
        return result;
      } catch (e) {
        this.error = e.message;
        throw e;
      } finally { this.uploading = false; }
    },
    async updateProfile(id, resumeInfo) {
      try {
        await api.put(`/profiles/${id}`, { resume_info: resumeInfo });
        await this.fetchProfile(id);
      } catch (e) { this.error = e.message; throw e; }
    },
    async deleteProfile(id) {
      try {
        await api.del(`/profiles/${id}`);
        this.profiles = this.profiles.filter(p => p.id !== id);
        // If last item on current page was deleted, go back a page
        if (this.profiles.length === 0 && this.page > 1) {
          this.page--;
        }
        await this.fetchProfiles(true);
      } catch (e) { this.error = e.message; throw e; }
    },
    async pollStatus(id, interval = 2000, maxPolls = 90) {
      const poll = async () => {
        try {
          const data = await api.get(`/profiles/${id}/status`);
          const numId = typeof id === 'string' ? parseInt(id) : id;
          const idx = this.profiles.findIndex(p => p.id === numId || p.id === id);
          if (idx !== -1) this.profiles[idx].status = data.status;
          if (this.current && (this.current.id === numId || this.current.id === id)) this.current.status = data.status;
          if (data.status === 'READY' || data.status === 'FAILED') return data.status;
          return null;
        } catch { return null; }
      };
      return new Promise((resolve) => {
        let count = 0;
        const timer = setInterval(async () => {
          count++;
          const result = await poll();
          if (result || count >= maxPolls) {
            clearInterval(timer);
            // Re-fetch full profiles list so resume_info is populated
            if (result === 'READY') this.fetchProfiles(true);
            resolve(result || 'FAILED');
          }
        }, interval);
      });
    },
  },
});

const useJobStore = defineStore('job', {
  state: () => ({
    jobs: [],
    current: null,
    customResume: null,
    selectedId: null,
    loading: false,
    error: null,
    page: 1,
    totalPages: 1,
    total: 0,
    limit: 10,
  }),
  actions: {
    selectJob(id) {
      this.selectedId = id;
    },
    async fetchJobs(force = false, { page, limit } = {}) {
      if (_jobFetchPromise) return _jobFetchPromise;
      if (!force && this.jobs.length > 0) return;
      const p = page || this.page;
      const l = limit || this.limit;
      this.loading = true;
      _jobFetchPromise = api.get(`/jobs/?page=${p}&limit=${l}`)
        .then(data => {
          this.jobs = data.items;
          this.total = data.total;
          this.page = data.page;
          this.totalPages = data.pages;
          this.limit = data.limit;
        })
        .catch(e => { this.error = e.message; })
        .finally(() => { this.loading = false; _jobFetchPromise = null; });
      return _jobFetchPromise;
    },
    async goToPage(n) {
      this.page = n;
      await this.fetchJobs(true, { page: n });
    },
    async fetchJob(id) {
      this.loading = true;
      // Only null state if loading a different job
      if (!this.current || this.current.id !== id) {
        this.current = null;
        this.customResume = null;
      }
      try {
        const data = await api.get(`/jobs/${id}`);
        this.current = data;
        this.customResume = data.custom_resume_data || null;
      } catch (e) { this.error = e.message; }
      finally { this.loading = false; }
    },
    async createJob(profileId, jobDescription) {
      this.error = null;
      try {
        return await api.post('/jobs/', { profile_id: profileId, job_description: jobDescription });
      } catch (e) { this.error = e.message; throw e; }
    },
    async generateResume(jobId) {
      try {
        return await api.post(`/jobs/${jobId}/generate-resume`);
      } catch (e) {
        // Check if it's a 429 (credit exhausted) via HTTP status code
        if (e.status === 429) {
          e.is429 = true;
        }
        this.error = e.message;
        throw e;
      }
    },
    async generatePdf(jobId) {
      try {
        return await api.post(`/jobs/${jobId}/generate-pdf`);
      } catch (e) { this.error = e.message; throw e; }
    },
    async pollStatus(id, interval = 2000, maxPolls = 90) {
      const terminalStatuses = ['READY', 'FAILED'];
      const poll = async () => {
        try {
          const data = await api.get(`/jobs/${id}/status`);
          const numId = typeof id === 'string' ? parseInt(id) : id;
          if (this.current && (this.current.id === numId || this.current.id === id)) this.current.status = data.status;
          const idx = this.jobs.findIndex(j => j.id === numId || j.id === id);
          if (idx !== -1) this.jobs[idx].status = data.status;
          if (terminalStatuses.includes(data.status)) return data.status;
          return null;
        } catch { return null; }
      };
      return new Promise((resolve) => {
        let count = 0;
        const timer = setInterval(async () => {
          count++;
          const result = await poll();
          if (result || count >= maxPolls) { clearInterval(timer); resolve(result || 'FAILED'); }
        }, interval);
      });
    },
  },
});

// ================================================================
// ROAST STORE
// ================================================================
let _roastFetchPromise = null;

const useRoastStore = defineStore('roast', {
  state: () => ({
    roasts: [],
    current: null,
    loading: false,
    uploading: false,
    error: null,
    page: 1,
    totalPages: 1,
    total: 0,
    limit: 10,
  }),
  actions: {
    async fetchRoasts(force = false, { page, limit } = {}) {
      if (_roastFetchPromise) return _roastFetchPromise;
      if (!force && this.roasts.length > 0) return;
      const p = page || this.page;
      const l = limit || this.limit;
      this.loading = true;
      _roastFetchPromise = api.get(`/roasts/?page=${p}&limit=${l}`)
        .then(data => {
          this.roasts = data.items;
          this.total = data.total;
          this.page = data.page;
          this.totalPages = data.pages;
          this.limit = data.limit;
        })
        .catch(e => { this.error = e.message; })
        .finally(() => { this.loading = false; _roastFetchPromise = null; });
      return _roastFetchPromise;
    },
    async uploadRoast(file) {
      this.uploading = true;
      this.error = null;
      try {
        return await api.upload('/roasts/upload', file);
      } catch (e) {
        this.error = e.message;
        throw e;
      } finally {
        this.uploading = false;
      }
    },
    async fetchRoast(id) {
      this.loading = true;
      this.current = null;
      try {
        this.current = await api.get(`/roasts/${id}`);
      } catch (e) { this.error = e.message; }
      finally { this.loading = false; }
    },
    async goToPage(n) {
      this.page = n;
      await this.fetchRoasts(true, { page: n });
    },
    async pollStatus(id, interval = 2000, maxPolls = 60) {
      const poll = async () => {
        try {
          const data = await api.get(`/roasts/${id}/status`);
          if (['READY', 'FAILED'].includes(data.status)) return data.status;
          return null;
        } catch { return null; }
      };
      return new Promise((resolve) => {
        let count = 0;
        const timer = setInterval(async () => {
          count++;
          const result = await poll();
          if (result || count >= maxPolls) { clearInterval(timer); resolve(result || 'FAILED'); }
        }, interval);
      });
    },
  },
});

// ================================================================
// CREDIT STORE
// ================================================================
const useCreditStore = defineStore('credit', {
  state: () => ({
    balance: 0,
    dailyFreeRemaining: 0,
    dailyFreeTotal: 3,
    activeTimePass: null,
    hasUnlimited: false,
    packs: [],
    timePasses: [],
    transactions: [],
    txnTotal: 0,
    txnPage: 1,
    txnPages: 1,
    loading: false,
    error: null,
  }),
  getters: {
    totalAvailable: (s) => s.hasUnlimited ? 999 : s.balance + s.dailyFreeRemaining,
    displayBalance: (s) => {
      if (s.hasUnlimited) return 'UNLIMITED';
      const total = s.balance + s.dailyFreeRemaining;
      return String(total);
    },
  },
  actions: {
    async fetchBalance() {
      try {
        const data = await api.get('/credits/me');
        this.balance = data.balance;
        this.dailyFreeRemaining = data.daily_free_remaining;
        this.dailyFreeTotal = data.daily_free_total;
        this.activeTimePass = data.active_time_pass;
        this.hasUnlimited = data.has_unlimited;
      } catch (e) { this.error = e.message; }
    },
    async fetchPacks() {
      try {
        const data = await api.get('/credits/packs');
        this.packs = data.credit_packs;
        this.timePasses = data.time_passes;
      } catch (e) { this.error = e.message; }
    },
    async fetchHistory(page = 1, search = '') {
      this.loading = true;
      try {
        const data = await api.get(`/credits/history?page=${page}&limit=20&search=${encodeURIComponent(search)}`);
        this.transactions = data.items;
        this.txnTotal = data.total;
        this.txnPage = data.page;
        this.txnPages = data.pages;
      } catch (e) { this.error = e.message; }
      this.loading = false;
    },
    async redeemPromo(code) {
      const data = await api.post('/credits/redeem-promo', { code });
      if (data.balance) {
        this.balance = data.balance.balance;
        this.dailyFreeRemaining = data.balance.daily_free_remaining;
        this.hasUnlimited = data.balance.has_unlimited;
        this.activeTimePass = data.balance.active_time_pass;
      }
      return data;
    },
  },
});

// ================================================================
// PAYWALL MODAL STATE
// ================================================================
const paywallState = reactive({
  visible: false,
  _resolve: null,
});

function showPaywall() {
  return new Promise((resolve) => {
    paywallState.visible = true;
    paywallState._resolve = resolve;
  });
}

function _closePaywall(result) {
  if (paywallState._resolve) paywallState._resolve(result);
  paywallState.visible = false;
  paywallState._resolve = null;
}

const PaywallModal = {
  template: `
    <Teleport to="body">
      <transition name="confirm">
        <div v-if="state.visible" class="confirm-backdrop" @click.self="close(false)">
          <div class="confirm-card" style="max-width:540px;border-color:rgba(251,65,0,0.2);">
            <div class="flex items-center gap-3 mb-5">
              <div class="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style="background:rgba(251,65,0,0.12);">
                <span style="font-size:22px;font-weight:800;color:var(--orange);font-family:var(--font-mono);">$</span>
              </div>
              <div>
                <h3 class="font-mono font-bold text-sm tracking-wide text-white">CREDITS REQUIRED</h3>
                <p class="text-[11px]" style="color:var(--text-dim)">Purchase credits to continue</p>
              </div>
            </div>

            <!-- Loading spinner -->
            <div v-if="packsLoading" class="flex justify-center py-6">
              <div class="loading-spinner"></div>
            </div>

            <!-- Credit Packs -->
            <div v-if="!packsLoading && creditStore.packs.length" class="mb-4">
              <div class="section-label mb-2">Credit Packs</div>
              <div class="space-y-2">
                <div v-for="p in creditStore.packs" :key="p.id"
                     class="flex items-center justify-between p-3 rounded-lg" style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.04);">
                  <div>
                    <span class="text-white text-sm font-semibold">{{ p.name }}</span>
                    <span class="text-xs ml-2" style="color:var(--teal)">{{ p.credits }} credits</span>
                  </div>
                  <button @click="buy('credit_pack', p.id, p.price_paise)" class="btn-ghost text-xs" :disabled="buying">
                    {{ formatPrice(p.price_paise) }}
                  </button>
                </div>
              </div>
            </div>

            <!-- Time Passes (Coming Soon) -->
            <div v-if="!packsLoading && creditStore.timePasses.length" class="mb-4 opacity-50">
              <div class="section-label mb-2">Time Passes <span class="text-[9px] font-mono ml-1 px-1.5 py-0.5 rounded" style="background:rgba(1,169,219,0.15);color:var(--teal);">COMING SOON</span></div>
              <div class="space-y-2">
                <div v-for="t in creditStore.timePasses" :key="t.id"
                     class="flex items-center justify-between p-3 rounded-lg" style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.04);">
                  <div>
                    <span class="text-white text-sm font-semibold">{{ t.name }}</span>
                    <span class="text-xs ml-2" style="color:var(--teal)">{{ t.duration_days }} days unlimited</span>
                  </div>
                  <span class="text-[10px] font-mono" style="color:var(--text-dim)">{{ formatPrice(t.price_paise) }}</span>
                </div>
              </div>
            </div>

            <!-- Promo Code -->
            <div class="mb-4">
              <div class="section-label mb-2">Promo Code</div>
              <div class="flex items-center gap-2">
                <input v-model="promoCode" placeholder="Enter code" class="input-field input-mono flex-1"
                       @keyup.enter="redeemPromo" :disabled="buying" />
                <button @click="redeemPromo" class="btn-ghost text-xs" :disabled="!promoCode.trim() || buying">REDEEM</button>
              </div>
              <p v-if="promoMsg" class="text-xs mt-2 font-mono" :style="{ color: promoError ? 'var(--red)' : 'var(--green)' }">{{ promoMsg }}</p>
            </div>

            <p v-if="error" class="text-red-400 text-xs mb-3 font-mono">{{ error }}</p>

            <div class="flex justify-end">
              <button @click="close(false)" class="btn-ghost">CLOSE</button>
            </div>
          </div>
        </div>
      </transition>
    </Teleport>
  `,
  setup() {
    const creditStore = useCreditStore();
    const buying = ref(false);
    const error = ref(null);
    const promoCode = ref('');
    const promoMsg = ref('');
    const promoError = ref(false);
    const packsLoading = ref(false);

    watch(() => paywallState.visible, async (v) => {
      if (v) {
        buying.value = false;
        error.value = null;
        promoCode.value = '';
        promoMsg.value = '';
        promoError.value = false;
        packsLoading.value = true;
        await creditStore.fetchPacks();
        packsLoading.value = false;
      }
    });

    // Close on Escape key
    function onEsc(e) { if (e.key === 'Escape' && paywallState.visible) close(false); }
    onMounted(() => window.addEventListener('keydown', onEsc));
    onUnmounted(() => window.removeEventListener('keydown', onEsc));

    function formatPrice(paise) {
      return '₹' + (paise / 100).toFixed(0);
    }

    async function buy(itemType, itemId, amountPaise) {
      if (typeof Razorpay === 'undefined') {
        error.value = 'Payment gateway not loaded. Please refresh the page.';
        return;
      }
      buying.value = true;
      error.value = null;
      try {
        const order = await api.post('/payments/create-order', { item_type: itemType, item_id: itemId });
        const options = {
          key: order.razorpay_key_id,
          amount: order.amount_paise,
          currency: order.currency,
          order_id: order.order_id,
          name: 'ATS Beater',
          description: itemType === 'credit_pack' ? 'Credit Pack' : 'Time Pass',
          handler: async function(response) {
            try {
              await api.post('/payments/verify', {
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature,
              });
              await creditStore.fetchBalance();
              close(true);
            } catch (e) {
              error.value = 'Payment verification failed: ' + e.message;
            }
            buying.value = false;
          },
          modal: {
            ondismiss: function() { buying.value = false; },
          },
          theme: { color: '#FB4100' },
        };
        const rzp = new Razorpay(options);
        rzp.open();
      } catch (e) {
        error.value = e.message;
        buying.value = false;
      }
    }

    async function redeemPromo() {
      if (!promoCode.value.trim() || buying.value) return;
      buying.value = true;
      promoMsg.value = '';
      promoError.value = false;
      try {
        const result = await creditStore.redeemPromo(promoCode.value);
        promoMsg.value = result.message;
        promoCode.value = '';
        // If credits were added, auto-close paywall after delay (guard against race)
        if (creditStore.totalAvailable > 0) {
          setTimeout(() => { if (paywallState.visible) close(true); }, 1000);
        }
      } catch (e) {
        promoMsg.value = e.message;
        promoError.value = true;
      }
      buying.value = false;
    }

    function close(result) {
      _closePaywall(result);
    }

    return { state: paywallState, creditStore, buying, error, promoCode, promoMsg, promoError, packsLoading, formatPrice, buy, redeemPromo, close };
  },
};

// ================================================================
// CONFIRM MODAL — replaces native window.confirm
// ================================================================
const confirmState = reactive({
  visible: false,
  title: '',
  message: '',
  confirmLabel: 'DELETE',
  variant: 'danger', // 'danger' | 'warning'
  _resolve: null,
});

function showConfirm({ title, message, confirmLabel = 'DELETE', variant = 'danger' }) {
  return new Promise((resolve) => {
    Object.assign(confirmState, { visible: true, title, message, confirmLabel, variant, _resolve: resolve });
  });
}

function _resolveConfirm(result) {
  if (confirmState._resolve) confirmState._resolve(result);
  confirmState.visible = false;
  confirmState._resolve = null;
}

const ConfirmModal = {
  template: `
    <Teleport to="body">
      <transition name="confirm">
        <div v-if="state.visible" class="confirm-backdrop" @click.self="cancel">
          <div class="confirm-card">
            <div class="flex items-center gap-3 mb-4">
              <div class="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                   :style="{ background: state.variant === 'warning' ? 'rgba(251,65,0,0.12)' : 'rgba(239,68,68,0.12)' }">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                     :stroke="state.variant === 'warning' ? 'var(--orange)' : 'var(--red)'"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <h3 class="font-mono font-bold text-sm tracking-wide text-white">{{ state.title }}</h3>
            </div>
            <p class="text-sm mb-6" style="color:var(--text-secondary);line-height:1.6;">{{ state.message }}</p>
            <div class="flex justify-end gap-3">
              <button @click="cancel" class="btn-ghost">{{ state.variant === 'warning' ? 'STAY' : 'CANCEL' }}</button>
              <button @click="ok" :class="state.variant === 'warning' ? 'btn-primary' : 'btn-danger'">{{ state.confirmLabel }}</button>
            </div>
          </div>
        </div>
      </transition>
    </Teleport>
  `,
  setup() {
    const cancel = () => _resolveConfirm(false);
    const ok = () => _resolveConfirm(true);
    const onKey = (e) => { if (e.key === 'Escape' && confirmState.visible) cancel(); };
    onMounted(() => document.addEventListener('keydown', onKey));
    onUnmounted(() => document.removeEventListener('keydown', onKey));
    return { state: confirmState, cancel, ok };
  },
};

// ================================================================
// COMPONENTS — Shared
// ================================================================
const StatusBadge = {
  props: ['status'],
  template: `
    <span :class="badgeClass" class="badge">{{ label }}</span>
  `,
  computed: {
    badgeClass() {
      const map = {
        PENDING: 'badge-pending', PROCESSING: 'badge-processing',
        READY: 'badge-ready', FAILED: 'badge-failed',
        GENERATING_RESUME: 'badge-processing', GENERATING_PDF: 'badge-processing',
        RESUME_GENERATED: 'badge-processing',
      };
      return map[this.status] || 'badge-pending';
    },
    label() {
      const map = {
        PENDING: 'Pending', PROCESSING: 'Processing',
        READY: 'Ready', FAILED: 'Failed',
        GENERATING_RESUME: 'Generating', GENERATING_PDF: 'Building PDF',
        RESUME_GENERATED: 'Building PDF',
      };
      return map[this.status] || this.status;
    },
  },
};

const PaginationControls = {
  props: ['page', 'totalPages'],
  emits: ['go'],
  template: `
    <div v-if="totalPages > 1" class="flex items-center justify-center gap-4 mt-6">
      <button class="btn-ghost text-xs" :disabled="page <= 1" @click="$emit('go', page - 1)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>
        PREV
      </button>
      <span class="font-mono text-xs text-slate-400">{{ page }} / {{ totalPages }}</span>
      <button class="btn-ghost text-xs" :disabled="page >= totalPages" @click="$emit('go', page + 1)">
        NEXT
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
      </button>
    </div>
  `,
};

// ================================================================
// COMPONENTS — TopHeader (global, receives hamburger via inject)
// ================================================================
const TopHeader = {
  template: `
    <header class="top-header">
      <div class="flex items-center min-w-0 flex-1">
        <button class="hamburger-btn" @click="toggle" aria-label="Open menu">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
        <div class="min-w-0">
          <slot name="left" />
        </div>
      </div>
      <div class="flex-shrink-0">
        <slot name="right" />
      </div>
    </header>
  `,
  setup() {
    const toggle = inject('toggleSidebar', () => {});
    return { toggle };
  },
};

// ================================================================
// COMPONENTS — App Shell
// ================================================================
const AppSidebar = {
  template: `
    <div class="flex flex-col h-full">
      <div class="px-5 py-5 border-b border-white/5">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg flex items-center justify-center" style="background: rgba(251,65,0,0.15);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FB4100" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 10V8a2.4 2.4 0 0 0-.706-1.704l-3.588-3.588A2.4 2.4 0 0 0 14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h4.35"/>
              <path d="M14 2v5a1 1 0 0 0 1 1h5"/>
              <path d="M16 14a2 2 0 0 0-2 2"/><path d="M16 22a2 2 0 0 1-2-2"/>
              <path d="M20 14a2 2 0 0 1 2 2"/><path d="M20 22a2 2 0 0 0 2-2"/>
            </svg>
          </div>
          <div>
            <div class="font-mono font-bold text-base tracking-tight" style="color:var(--orange)">ATS BEATER</div>
            <div class="text-[10px] font-semibold tracking-widest uppercase" style="color:var(--text-dim)">Resume Tailoring</div>
          </div>
        </div>
      </div>
      <nav class="flex-1 py-4 px-3 space-y-1">
        <router-link to="/dashboard" class="sidebar-link" active-class="active">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="4" rx="1"/><rect x="14" y="10" width="7" height="7" rx="1"/><rect x="3" y="13" width="7" height="4" rx="1"/></svg>
          Dashboard
        </router-link>
        <router-link to="/profiles" class="sidebar-link" active-class="active">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
          Profiles
        </router-link>
        <router-link to="/jobs" class="sidebar-link" active-class="active">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          Jobs
        </router-link>
        <div class="pt-4 mt-4 border-t border-white/5">
          <router-link to="/profiles/new" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4v16m-8-8h16"/></svg>
            New Profile
          </router-link>
          <router-link to="/jobs/new" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            New Job
          </router-link>
        </div>
        <div class="pt-4 mt-4 border-t border-white/5">
          <router-link to="/roast" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22c-4 0-8-2-8-8 0-4 2-6 4-8 0 3 1.5 4 3 4-1-3 1-7 5-10 0 4 2 6 3 7 1.5 1.5 2 3 2 5 0 6-4 10-9 10z"/></svg>
            Resume Roast
          </router-link>
          <router-link to="/credits" class="sidebar-link" active-class="active">
            <span style="font-size:16px;font-weight:700;width:18px;text-align:center;display:inline-block;">$</span>
            Credits
          </router-link>
        </div>
        <div class="pt-4 mt-4 border-t border-white/5">
          <router-link to="/settings/ai" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12.22 2h-.44a2 2 0 00-2 2v.18a2 2 0 01-1 1.73l-.43.25a2 2 0 01-2 0l-.15-.08a2 2 0 00-2.73.73l-.22.38a2 2 0 00.73 2.73l.15.1a2 2 0 011 1.72v.51a2 2 0 01-1 1.74l-.15.09a2 2 0 00-.73 2.73l.22.38a2 2 0 002.73.73l.15-.08a2 2 0 012 0l.43.25a2 2 0 011 1.73V20a2 2 0 002 2h.44a2 2 0 002-2v-.18a2 2 0 011-1.73l.43-.25a2 2 0 012 0l.15.08a2 2 0 002.73-.73l.22-.39a2 2 0 00-.73-2.73l-.15-.08a2 2 0 01-1-1.74v-.5a2 2 0 011-1.74l.15-.09a2 2 0 00.73-2.73l-.22-.38a2 2 0 00-2.73-.73l-.15.08a2 2 0 01-2 0l-.43-.25a2 2 0 01-1-1.73V4a2 2 0 00-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
            AI Settings
          </router-link>
        </div>
        <div class="pt-4 mt-4 border-t border-white/5" v-if="auth.isSuperAdmin">
          <router-link to="/admin" class="sidebar-link" active-class="active">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
            Admin
          </router-link>
        </div>
      </nav>
      <!-- Credit Badge -->
      <div v-if="auth.user" class="mx-3 mb-2 p-3 rounded-lg" style="background:rgba(251,65,0,0.06);border:1px solid rgba(251,65,0,0.15);">
        <div class="flex items-center justify-between mb-1">
          <span class="text-[9px] font-mono font-bold tracking-widest uppercase" style="color:var(--orange)">Credits</span>
          <span class="font-mono font-bold text-sm" style="color:var(--orange)">{{ creditStore.displayBalance }}</span>
        </div>
        <div class="text-[10px] font-mono" style="color:var(--text-dim)">
          {{ creditStore.dailyFreeRemaining }}/{{ creditStore.dailyFreeTotal }} free today
        </div>
        <router-link v-if="creditStore.totalAvailable === 0" to="/credits"
                     class="block mt-2 text-center text-[10px] font-mono font-bold tracking-wide py-1.5 rounded"
                     style="background:var(--orange);color:white;">
          GET MORE CREDITS
        </router-link>
      </div>
      <div class="px-4 py-4 border-t border-white/5" v-if="auth.user">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
               style="background:linear-gradient(135deg, var(--orange), rgba(251,65,0,0.4)); color:white;">
            {{ auth.initials }}
          </div>
          <div class="flex-1 min-w-0">
            <div class="text-sm font-medium truncate text-white">{{ auth.user.name }}</div>
            <div class="text-[11px] truncate" style="color:var(--text-dim)">{{ auth.user.email }}</div>
            <div v-if="auth.tenantName" class="text-[10px] truncate font-mono" style="color:var(--teal)">{{ auth.tenantName }}</div>
          </div>
          <button @click="doLogout" class="text-slate-500 hover:text-red-400 transition" title="Logout">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/></svg>
          </button>
        </div>
      </div>
    </div>
  `,
  setup() {
    const auth = useAuthStore();
    const creditStore = useCreditStore();
    const router = useRouter();
    onMounted(() => { if (auth.user) creditStore.fetchBalance(); });
    const doLogout = () => {
      auth.logout();
      creditStore.$reset();
      router.push('/login');
    };
    return { auth, creditStore, doLogout };
  },
};

// ================================================================
// PAGES — Login
// ================================================================
const LoginPage = {
  template: `
    <div class="flex items-center justify-center h-screen relative z-10">
      <div class="w-full max-w-md px-4 md:px-0">
        <!-- Boot sequence -->
        <div class="mb-8 px-4" v-if="bootDone">
          <div v-for="(line, i) in bootLines" :key="i"
               class="boot-line" :class="{ visible: line.visible, ok: line.ok, warn: line.warn }">
            {{ line.text }}
          </div>
        </div>
        <!-- Login card -->
        <div class="widget-card fade-slide-in" :class="{ 'stagger-3': bootDone }" v-if="showCard">
          <div class="text-center mb-8">
            <div class="flex items-center justify-center gap-3 mb-4">
              <div class="w-12 h-12 rounded-lg flex items-center justify-center" style="background: rgba(251,65,0,0.15);">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FB4100" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M20 10V8a2.4 2.4 0 0 0-.706-1.704l-3.588-3.588A2.4 2.4 0 0 0 14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h4.35"/>
                  <path d="M14 2v5a1 1 0 0 0 1 1h5"/>
                  <path d="M16 14a2 2 0 0 0-2 2"/><path d="M16 22a2 2 0 0 1-2-2"/>
                  <path d="M20 14a2 2 0 0 1 2 2"/><path d="M20 22a2 2 0 0 0 2-2"/>
                </svg>
              </div>
            </div>
            <h1 class="font-mono text-2xl font-bold text-white tracking-tight">ATS BEATER</h1>
            <p class="font-mono text-[10px] tracking-widest uppercase mt-1" style="color:var(--text-dim)">AI-Powered Resume Tailoring</p>
          </div>
          <div class="space-y-4">
            <button @click="loginGoogle" class="w-full justify-center"
                    style="background: rgba(255,255,255,0.95); border: 1px solid rgba(255,255,255,0.1); color: #1f2937; padding: 14px 28px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 13px; letter-spacing: 1.5px; cursor: pointer; transition: all 0.3s ease; text-transform: uppercase; display: inline-flex; align-items: center; gap: 10px; box-shadow: 0 4px 20px rgba(1,169,219,0.15);"
                    onmouseover="this.style.background='white'; this.style.boxShadow='0 6px 30px rgba(1,169,219,0.25)'; this.style.transform='translateY(-2px)';"
                    onmouseout="this.style.background='rgba(255,255,255,0.95)'; this.style.boxShadow='0 4px 20px rgba(1,169,219,0.15)'; this.style.transform='none';">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
              SIGN IN WITH GOOGLE
            </button>
          </div>
          <p class="text-center text-[11px] mt-6" style="color:var(--text-dim)">
            By signing in you consent to the storage of your resume data for processing.
          </p>
        </div>
      </div>
    </div>
  `,
  setup() {
    const bootLines = ref([
      { text: '[SYS] Initializing resume engine...', visible: false, ok: false },
      { text: '[SYS] Loading AI inference module... OK', visible: false, ok: true },
      { text: '[SYS] LaTeX compiler found at /Library/TeX/texbin', visible: false, ok: true },
      { text: '[SYS] PostgreSQL connection... OK', visible: false, ok: true },
      { text: '[SYS] Awaiting authentication...', visible: false, warn: true },
    ]);
    const bootDone = ref(true);
    const showCard = ref(false);

    onMounted(() => {
      // Token handling is done in the router guard — LoginPage only shows
      // when the user genuinely needs to log in (no valid token).
      // Boot sequence animation
      bootLines.value.forEach((line, i) => {
        setTimeout(() => { line.visible = true; }, 200 * (i + 1));
      });
      setTimeout(() => { showCard.value = true; }, 200 * bootLines.value.length + 300);
    });

    const loginGoogle = async () => {
      try {
        const data = await api.get('/auth/google/login');
        window.location.href = data.url;
      } catch (e) {
        alert('Failed to start login: ' + e.message);
      }
    };
    return { bootLines, bootDone, showCard, loginGoogle };
  },
};

// ================================================================
// PAGES — Dashboard
// ================================================================
const DashboardPage = {
  components: { StatusBadge },
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div v-if="dataReady && isNewUser">
            <h1 class="text-sm font-bold text-white font-mono tracking-tight">WELCOME</h1>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Let's get your resume job-ready.</p>
          </div>
          <div v-else>
            <div class="flex items-center gap-2">
              <h1 class="text-sm font-bold text-white font-mono tracking-tight">OPERATIONS CENTER</h1>
              <span class="badge badge-live">LIVE</span>
            </div>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Resume Builder Command</p>
          </div>
        </template>
        <template #right>
          <div v-if="!(dataReady && isNewUser)" class="hidden md:flex items-center gap-3">
            <router-link to="/roast" class="btn-ghost" style="color:var(--orange); border-color:rgba(249,115,22,0.3);">ROAST MY RESUME</router-link>
            <router-link to="/profiles/new" class="btn-ghost">+ NEW PROFILE</router-link>
            <router-link to="/jobs/new" class="btn-ghost">+ NEW JOB</router-link>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">

        <!-- Loading gate — wait for stores before deciding which view -->
        <div v-if="!dataReady" class="flex items-center justify-center py-16">
          <div class="spinner"></div>
        </div>

        <!-- ===== MODE 1: First-Time Onboarding ===== -->
        <template v-else-if="isNewUser">
          <!-- Hero: Roast CTA -->
          <div class="widget-card fade-slide-in stagger-1 mb-8" style="border-left:3px solid var(--orange); box-shadow:0 0 24px rgba(249,115,22,0.08);">
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4 p-2">
              <div>
                <div class="flex items-center gap-2 mb-2">
                  <span class="text-2xl">&#x1F525;</span>
                  <span class="text-xs font-mono font-bold tracking-widest" style="color:var(--orange)">START HERE</span>
                </div>
                <p class="text-white text-sm font-medium mb-1">Get a brutally honest AI roast of your resume in 30 seconds.</p>
                <p class="text-xs" style="color:var(--text-dim)">Free, instant, no sign-up needed.</p>
              </div>
              <router-link to="/roast" class="btn-primary whitespace-nowrap text-center" style="min-width:180px;">ROAST MY RESUME</router-link>
            </div>
          </div>

          <!-- How It Works -->
          <div class="fade-slide-in stagger-2 mb-8">
            <div class="section-label mb-4">HOW IT WORKS</div>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div v-for="(step, i) in onboardingSteps" :key="i" class="flex items-start gap-3">
                <span class="font-mono text-lg font-bold flex-shrink-0" style="color:var(--orange); opacity:0.7;">{{ i + 1 }}</span>
                <div>
                  <p class="text-white text-xs font-bold font-mono tracking-wide mb-0.5">{{ step.title }}</p>
                  <p class="text-[11px] leading-relaxed" style="color:var(--text-dim)">{{ step.desc }}</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Secondary CTAs -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 fade-slide-in stagger-3">
            <div class="widget-card p-5">
              <p class="text-white text-xs font-bold font-mono tracking-wide mb-2">UPLOAD RESUME</p>
              <p class="text-[11px] mb-4" style="color:var(--text-dim)">Create a structured profile from your PDF. Edit and reuse it across multiple job applications.</p>
              <router-link to="/profiles/new" class="btn-secondary text-xs inline-block">UPLOAD PDF</router-link>
            </div>
            <div class="widget-card p-5">
              <p class="text-white text-xs font-bold font-mono tracking-wide mb-2">SKIP TO JOB</p>
              <p class="text-[11px] mb-4" style="color:var(--text-dim)">Already have a profile? Paste a job description and let AI tailor your resume instantly.</p>
              <router-link to="/jobs/new" class="btn-secondary text-xs inline-block">CREATE JOB</router-link>
            </div>
          </div>
        </template>

        <!-- ===== MODE 2: Returning User Dashboard ===== -->
        <template v-else>
          <!-- KPI Cards -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div class="widget-card fade-slide-in stagger-1 text-center">
              <p class="kpi-label mb-2">Total Profiles</p>
              <p class="kpi-value">{{ profileStore.total }}</p>
              <p class="text-xs mt-1" style="color:var(--text-dim)">{{ totalProfiles }} total</p>
            </div>
            <div class="widget-card fade-slide-in stagger-2 text-center">
              <p class="kpi-label mb-2">Total Jobs</p>
              <p class="kpi-value">{{ jobStore.total }}</p>
              <p class="text-xs mt-1" style="color:var(--text-dim)">{{ totalJobs }} total</p>
            </div>
            <div class="widget-card fade-slide-in stagger-3 text-center">
              <p class="kpi-label mb-2">Credits Available</p>
              <p class="kpi-value" style="color:var(--orange)">{{ creditStore.displayBalance }}</p>
              <p v-if="!creditStore.hasUnlimited" class="text-xs mt-1" style="color:var(--text-dim)">{{ creditStore.dailyFreeRemaining }} free today</p>
              <p v-else class="text-xs mt-1" style="color:var(--teal)">Time pass active</p>
            </div>
          </div>

          <!-- Nudge Banner -->
          <div v-if="showNudge" class="widget-card fade-slide-in mb-6" style="border-left:3px solid var(--teal);">
            <div class="flex items-center justify-between gap-4 p-1">
              <div class="flex items-center gap-3">
                <span class="text-lg">&#x1F4A1;</span>
                <p class="text-xs text-white">Your profile is ready — <span style="color:var(--text-dim)">paste a job description to get a tailored resume.</span></p>
              </div>
              <div class="flex items-center gap-3 flex-shrink-0">
                <router-link to="/jobs/new" class="btn-primary text-xs">CREATE JOB</router-link>
                <button @click="nudgeDismissed = true" class="text-xs" style="color:var(--text-dim); cursor:pointer; background:none; border:none; padding:4px;">&#x2715;</button>
              </div>
            </div>
          </div>

          <!-- Two columns -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Profiles -->
            <div class="fade-slide-in stagger-4">
              <div class="section-label">Recent Profiles</div>
              <div class="widget-card">
                <div v-if="profileStore.loading" class="flex items-center justify-center py-8">
                  <div class="spinner"></div>
                </div>
                <div v-else-if="profiles.length === 0" class="empty-state">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                  <p class="text-sm">No profiles yet</p>
                  <router-link to="/profiles/new" class="btn-secondary mt-3 inline-block text-xs">New Profile</router-link>
                </div>
                <table v-else class="data-table">
                  <thead><tr><th>Profile</th><th>Status</th><th>Date</th></tr></thead>
                  <tbody>
                    <tr v-for="p in profiles" :key="p.id" @click="goProfile(p.id)">
                      <td class="text-white font-medium">{{ p.resume_info?.name || 'Untitled' }}</td>
                      <td><StatusBadge :status="p.status" /></td>
                      <td class="font-mono text-[11px]">{{ formatDate(p.created_at) }}</td>
                    </tr>
                  </tbody>
                </table>
                <div v-if="profileStore.total > 5" class="pt-3 text-center">
                  <router-link to="/profiles" class="text-xs font-mono" style="color:var(--teal)">VIEW ALL ({{ profileStore.total }})</router-link>
                </div>
              </div>
            </div>
            <!-- Jobs -->
            <div class="fade-slide-in stagger-5">
              <div class="section-label">Recent Jobs</div>
              <div class="widget-card">
                <div v-if="jobStore.loading" class="flex items-center justify-center py-8">
                  <div class="spinner"></div>
                </div>
                <div v-else-if="jobs.length === 0" class="empty-state">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                  <p class="text-sm">No jobs yet</p>
                  <router-link to="/jobs/new" class="btn-secondary mt-3 inline-block text-xs">Create Job</router-link>
                </div>
                <table v-else class="data-table">
                  <thead><tr><th>Job</th><th>Status</th><th>Date</th></tr></thead>
                  <tbody>
                    <tr v-for="j in jobs" :key="j.id" @click="goJob(j.id)">
                      <td class="text-white font-medium">{{ jobLabel(j) }}</td>
                      <td><StatusBadge :status="j.status" /></td>
                      <td class="font-mono text-[11px]">{{ formatDate(j.created_at) }}</td>
                    </tr>
                  </tbody>
                </table>
                <div v-if="jobStore.total > 5" class="pt-3 text-center">
                  <router-link to="/jobs" class="text-xs font-mono" style="color:var(--teal)">VIEW ALL ({{ jobStore.total }})</router-link>
                </div>
              </div>
            </div>
          </div>
        </template>

      </div>
    </div>
  `,
  setup() {
    const router = useRouter();
    const profileStore = useProfileStore();
    const jobStore = useJobStore();
    const creditStore = useCreditStore();
    const roastStore = useRoastStore();
    const dataReady = ref(false);
    onMounted(async () => {
      await Promise.all([
        profileStore.fetchProfiles(true, { limit: 5 }),
        jobStore.fetchJobs(true, { limit: 5 }),
        creditStore.fetchBalance(),
        roastStore.fetchRoasts(true),
      ]);
      dataReady.value = true;
    });
    const profiles = computed(() => profileStore.profiles);
    const jobs = computed(() => jobStore.jobs);
    const totalProfiles = computed(() => profileStore.total || 0);
    const totalJobs = computed(() => jobStore.total || 0);
    const isNewUser = computed(() =>
      profileStore.total === 0 && jobStore.total === 0 && roastStore.total === 0
    );
    const nudgeDismissed = ref(false);
    const showNudge = computed(() =>
      profiles.value.length > 0 && jobStore.total === 0 && !nudgeDismissed.value
    );
    const onboardingSteps = [
      { title: 'ROAST', desc: 'Get a brutally honest score and breakdown of your resume.' },
      { title: 'UPLOAD', desc: 'Create a structured profile from your PDF resume.' },
      { title: 'TAILOR', desc: 'Paste a job description — AI customizes your resume.' },
      { title: 'DOWNLOAD', desc: 'Get a polished, ATS-optimized PDF ready to submit.' },
    ];
    const goProfile = (id) => { profileStore.selectProfile(id); router.push('/profiles/view'); };
    const goJob = (id) => { jobStore.selectJob(id); router.push('/jobs/view'); };
    return {
      profileStore, jobStore, creditStore, roastStore,
      profiles, jobs, totalProfiles, totalJobs,
      isNewUser, dataReady, nudgeDismissed, showNudge, onboardingSteps,
      formatDate, goProfile, goJob, jobLabel,
    };
  },
};

// ================================================================
// PAGES — Profile List
// ================================================================
const ProfileListPage = {
  components: { StatusBadge, PaginationControls },
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <h1 class="text-sm font-bold text-white font-mono tracking-tight">PROFILES</h1>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Manage your resume profiles</p>
          </div>
        </template>
        <template #right>
          <router-link to="/profiles/new" class="btn-ghost">+ NEW PROFILE</router-link>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div v-if="store.loading && store.profiles.length === 0" class="flex items-center justify-center py-16"><div class="spinner"></div></div>
        <div v-else-if="store.profiles.length === 0" class="empty-state py-16">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
          <p class="text-lg font-semibold text-white mt-2">No profiles yet</p>
          <p class="text-sm mb-4">Create a profile from your resume to get started</p>
          <router-link to="/profiles/new" class="btn-primary">NEW PROFILE</router-link>
        </div>
        <template v-else>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div v-for="p in store.profiles" :key="p.id"
                 class="widget-card cursor-pointer fade-slide-in"
                 @click="goProfile(p.id)">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-white font-bold text-sm">
                  {{ p.resume_info?.name || 'Untitled Profile' }}
                </h3>
                <StatusBadge :status="p.status" />
              </div>
              <p class="text-xs text-slate-400 mb-3" v-if="p.resume_info?.email">
                {{ p.resume_info.email }}
              </p>
              <div class="text-[10px] font-mono text-slate-600">
                <span>{{ formatDate(p.created_at) }}</span>
              </div>
            </div>
          </div>
          <PaginationControls :page="store.page" :totalPages="store.totalPages" @go="store.goToPage" />
        </template>
      </div>
    </div>
  `,
  setup() {
    const router = useRouter();
    const store = useProfileStore();
    onMounted(() => store.fetchProfiles(true, { limit: 10 }));
    const goProfile = (id) => { store.selectProfile(id); router.push('/profiles/view'); };
    return { store, formatDate: (iso) => formatDate(iso, { includeYear: true }), goProfile };
  },
};

// ================================================================
// PAGES — Profile Upload
// ================================================================
const ProfileUploadPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <div class="flex items-center gap-2 text-sm font-mono">
              <router-link to="/profiles" class="font-bold hover:text-white transition" style="color:var(--text-dim)">PROFILES</router-link>
              <span style="color:var(--text-dim)">/</span>
              <span class="font-bold text-white">New</span>
            </div>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Create a new profile from your PDF resume</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div class="max-w-xl mx-auto">
          <!-- Loading check -->
          <div v-if="checkingProfiles" class="flex items-center justify-center py-16"><div class="spinner"></div></div>
          <!-- Already processing -->
          <div v-else-if="processingProfile && !uploading && !uploadResult" class="widget-card text-center py-12 fade-slide-in">
            <div class="mb-4">
              <div class="spinner mx-auto" style="width:32px;height:32px;border-width:2px;"></div>
            </div>
            <p class="text-white font-semibold mb-1">A profile is already being processed</p>
            <p class="text-xs mb-4" style="color:var(--text-dim)">
              Your previous upload is still being analyzed. Please wait for it to finish before uploading another one.
            </p>
            <div class="mx-auto mb-6 px-4 py-3 rounded-lg text-left" style="background:rgba(251,65,0,0.08); border:1px solid rgba(251,65,0,0.2); max-width:360px;">
              <div class="flex items-start gap-2">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2" class="flex-shrink-0 mt-0.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <div>
                  <p class="text-xs font-semibold" style="color:var(--orange)">Usually 1-2 minutes</p>
                  <p class="text-[10px] mt-0.5" style="color:var(--text-dim)">Can take up to 3-4 min in rare cases. If it fails, we will let you know.</p>
                </div>
              </div>
            </div>
            <button @click="goToProcessing" class="btn-primary">VIEW PROGRESS</button>
          </div>
          <!-- Upload zone -->
          <div v-else-if="!uploading && !uploadResult" class="fade-slide-in">
            <div class="upload-zone mb-6"
                 :class="{ dragover: isDragging }"
                 @dragover.prevent="isDragging = true"
                 @dragleave="isDragging = false"
                 @drop.prevent="handleDrop"
                 @click="$refs.fileInput.click()">
              <input type="file" ref="fileInput" accept=".pdf" class="hidden" @change="handleFileSelect">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" class="mx-auto mb-4" style="color:var(--teal); opacity:0.6;">
                <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
              </svg>
              <p class="text-white font-semibold mb-1" v-if="!selectedFile">Drop your PDF here or click to browse</p>
              <p class="text-white font-semibold mb-1" v-else>{{ selectedFile.name }}</p>
              <p class="text-xs" style="color:var(--text-dim)">PDF files only, max 5MB</p>
            </div>
            <!-- Consent -->
            <div class="consent-check mb-6" :class="{ checked: consentAccepted }" @click="consentAccepted = !consentAccepted">
              <div class="check-box">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" :style="{ opacity: consentAccepted ? 1 : 0 }">
                  <path d="M2.5 6L5 8.5L9.5 3.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
              </div>
              <span class="text-sm" :style="{ color: consentAccepted ? 'var(--text-primary)' : 'var(--text-secondary)' }">
                I consent to the storage and AI processing of my resume data (name, email, phone) for generating customized resumes.
              </span>
            </div>
            <!-- Upload button -->
            <button class="btn-primary w-full justify-center" @click="doUpload"
                    :disabled="!selectedFile || !consentAccepted">
              CREATE PROFILE
            </button>
            <p v-if="error" class="text-red-400 text-xs mt-3 text-center font-mono">{{ error }}</p>
          </div>
          <!-- Processing state -->
          <div v-if="uploading" class="fade-slide-in">
            <!-- Status header -->
            <div class="widget-card mb-4">
              <div class="flex items-center gap-3 mb-3">
                <div class="spinner" style="width:20px;height:20px;border-width:2px;"></div>
                <p class="text-white font-semibold text-sm font-mono">{{ phaseLabel }}</p>
              </div>
              <div class="progress-track" style="max-width:100%;">
                <div class="progress-fill transition-all duration-700" :style="{ width: progress + '%' }"></div>
              </div>
              <div class="mt-3 space-y-1">
                <div v-for="(line, i) in statusLines" :key="i" class="boot-line visible" :class="{ ok: line.ok }">
                  {{ line.text }}
                </div>
              </div>
            </div>
            <!-- OCR text cycling — "reading your resume" -->
            <div v-if="ocrLines.length > 0 && !ocrDone" class="widget-card" style="border-left: 2px solid var(--teal);">
              <div class="flex items-center gap-2 mb-3">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                <span class="text-[10px] font-mono font-bold tracking-widest" style="color:var(--teal)">READING RESUME</span>
              </div>
              <div class="overflow-hidden relative flex flex-col justify-end" style="height:180px;">
                <div class="space-y-0.5">
                  <div v-for="(line, i) in visibleOcrLines" :key="i"
                       class="font-mono text-[11px] leading-relaxed transition-opacity duration-300"
                       :style="{ color: i === visibleOcrLines.length - 1 ? 'var(--text-primary)' : 'var(--text-dim)', opacity: i < visibleOcrLines.length - 3 ? 0.3 : (i < visibleOcrLines.length - 1 ? 0.6 : 1) }">
                    <span v-if="i === visibleOcrLines.length - 1" style="color:var(--teal);">&gt; </span>{{ line }}
                  </div>
                </div>
              </div>
            </div>
            <!-- AI structuring — shown after OCR cycling finishes while AI still works -->
            <div v-if="ocrDone" class="widget-card fade-slide-in" style="border-left: 2px solid var(--orange);">
              <div class="flex items-center gap-3">
                <div class="spinner" style="width:18px;height:18px;border-width:2px;border-color:rgba(251,65,0,0.2);border-top-color:var(--orange);"></div>
                <div>
                  <p class="text-sm font-mono font-semibold" style="color:var(--orange)">STRUCTURING WITH AI</p>
                  <p class="text-[10px] font-mono mt-0.5" style="color:var(--text-dim)">Extraction complete — organizing your resume data...</p>
                </div>
              </div>
              <div class="mt-3 px-3 py-2 rounded" style="background:rgba(251,65,0,0.06); border:1px solid rgba(251,65,0,0.15);">
                <div class="flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2" class="flex-shrink-0"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  <p class="text-[11px] font-mono" style="color:var(--orange)">Usually 1-2 min, up to 3-4 min in rare cases. Please stay on this page.</p>
                </div>
              </div>
            </div>
          </div>
          <!-- Done -->
          <div v-if="uploadResult && !uploading" class="widget-card text-center py-12 fade-slide-in">
            <div v-if="finalStatus === 'READY'" class="mb-4">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="1.5" class="mx-auto">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <div v-else class="mb-4">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5" class="mx-auto">
                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <p class="text-white font-semibold mb-1">
              {{ finalStatus === 'READY' ? 'Resume processed successfully!' : 'Processing failed' }}
            </p>
            <p class="text-xs mb-6" style="color:var(--text-dim)">
              {{ finalStatus === 'READY' ? 'Your profile is ready. You can now create jobs.' : 'Please try again or upload a different file.' }}
            </p>
            <div class="flex items-center justify-center gap-3">
              <button v-if="finalStatus === 'READY'" @click="goToProfile" class="btn-primary">
                VIEW PROFILE
              </button>
              <button @click="reset" class="btn-secondary">
                {{ finalStatus === 'READY' ? 'CREATE ANOTHER' : 'TRY AGAIN' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const router = useRouter();
    const store = useProfileStore();
    const selectedFile = ref(null);
    const consentAccepted = ref(false);
    const isDragging = ref(false);
    const uploading = ref(false);
    const uploadResult = ref(null);
    const finalStatus = ref(null);
    const error = ref(null);
    const progress = ref(0);
    const statusLines = ref([]);
    const phaseLabel = ref('Uploading...');
    const processingProfile = ref(null);
    const checkingProfiles = ref(true);
    let unmounted = false;

    onMounted(async () => {
      try {
        await store.fetchProfiles(true);
        const inProgress = store.profiles.find(
          p => p.status === 'PENDING' || p.status === 'PROCESSING'
        );
        if (inProgress) processingProfile.value = inProgress;
      } finally {
        checkingProfiles.value = false;
      }
    });

    // OCR text cycling
    const ocrLines = ref([]);
    const visibleOcrLines = ref([]);
    const ocrDone = ref(false);  // true once all lines have been shown
    let ocrTimer = null;
    let progressTimerRef = null;

    const MAX_VISIBLE_LINES = 9;

    const startOcrCycling = (text) => {
      // Split on newlines, filter empties, trim
      const lines = text.split(/\n/).map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length === 0) return;
      ocrLines.value = lines;
      ocrDone.value = false;
      let idx = 0;
      // Show first line immediately
      visibleOcrLines.value = [lines[0]];
      idx = 1;
      // Adaptive speed — target ~20s total for a relaxed feel
      const delay = Math.max(150, Math.min(350, 20000 / lines.length));
      ocrTimer = setInterval(() => {
        if (idx < lines.length) {
          // Sliding window: keep only the last N lines visible
          const next = [...visibleOcrLines.value, lines[idx]];
          visibleOcrLines.value = next.length > MAX_VISIBLE_LINES ? next.slice(-MAX_VISIBLE_LINES) : next;
          idx++;
        } else {
          clearInterval(ocrTimer);
          ocrTimer = null;
          ocrDone.value = true;
        }
      }, delay);
    };

    const stopOcrCycling = () => {
      if (ocrTimer) { clearInterval(ocrTimer); ocrTimer = null; }
      if (progressTimerRef) { clearInterval(progressTimerRef); progressTimerRef = null; }
    };

    const handleDrop = (e) => {
      isDragging.value = false;
      const file = e.dataTransfer.files[0];
      if (file && file.type === 'application/pdf') selectedFile.value = file;
    };
    const handleFileSelect = (e) => {
      const file = e.target.files[0];
      if (file) selectedFile.value = file;
    };
    const doUpload = async () => {
      if (!selectedFile.value || !consentAccepted.value) return;
      if (selectedFile.value.type !== 'application/pdf') {
        error.value = 'Only PDF files are accepted.';
        return;
      }
      if (selectedFile.value.size > 5 * 1024 * 1024) {
        error.value = 'File size must be under 5MB. Try exporting a smaller version without embedded images.';
        return;
      }
      uploading.value = true;
      error.value = null;
      progress.value = 10;
      phaseLabel.value = 'Uploading PDF...';
      statusLines.value = [{ text: '[SYS] Uploading PDF to server...', ok: false }];

      try {
        const result = await store.uploadProfile(selectedFile.value);
        uploadResult.value = result;

        // Phase 1 complete — upload + OCR extraction done server-side
        progress.value = 30;
        phaseLabel.value = 'Reading your resume...';
        statusLines.value = [
          { text: '[SYS] Upload complete', ok: true },
          { text: '[OCR] Text extracted from PDF', ok: true },
          { text: '[AI] Structuring resume with AI...', ok: false },
        ];

        // Start cycling through the extracted text
        if (result.extracted_text) {
          startOcrCycling(result.extracted_text);
        }

        // Smooth progress while polling
        progress.value = 40;
        progressTimerRef = setInterval(() => {
          if (progress.value < 85) progress.value += 2;
        }, 1500);

        // Poll for AI completion
        const status = await store.pollStatus(result.profile_id);
        clearInterval(progressTimerRef);
        progressTimerRef = null;
        stopOcrCycling();
        if (unmounted) return;

        // Show last window of OCR lines
        if (ocrLines.value.length > 0) {
          visibleOcrLines.value = ocrLines.value.slice(-MAX_VISIBLE_LINES);
        }

        finalStatus.value = status;
        progress.value = 100;

        if (status === 'READY') {
          // Go straight to the profile — no intermediate "done" screen
          phaseLabel.value = 'Profile ready!';
          statusLines.value.push({ text: '[SYS] AI structuring complete!', ok: true });
          await new Promise(r => setTimeout(r, 600));
          if (unmounted) return;
          store.selectProfile(result.profile_id);
          uploading.value = false;
          router.push('/profiles/view');
          return;
        }

        // Only show the "done" card for failures
        phaseLabel.value = 'Processing failed';
        statusLines.value.push({ text: '[SYS] Processing failed', ok: false });
        await new Promise(r => setTimeout(r, 500));
        uploading.value = false;
      } catch (e) {
        stopOcrCycling();
        if (unmounted) return;
        error.value = e.message;
        uploading.value = false;
      }
    };

    const reset = () => {
      selectedFile.value = null;
      consentAccepted.value = false;
      uploading.value = false;
      uploadResult.value = null;
      finalStatus.value = null;
      error.value = null;
      progress.value = 0;
      statusLines.value = [];
      ocrLines.value = [];
      visibleOcrLines.value = [];
      ocrDone.value = false;
      stopOcrCycling();
    };
    const goToProfile = () => {
      if (uploadResult.value) {
        store.selectProfile(uploadResult.value.profile_id);
        router.push('/profiles/view');
      }
    };
    const goToProcessing = () => {
      store.selectProfile(processingProfile.value.id);
      router.push('/profiles/view');
    };

    onUnmounted(() => { unmounted = true; stopOcrCycling(); });

    onBeforeRouteLeave(async () => {
      if (!uploading.value) return true;
      const leave = await showConfirm({
        title: 'Processing in Progress',
        message: 'Your profile is still being processed. It will continue in the background if you leave, but you won\'t see the live progress.',
        confirmLabel: 'LEAVE',
        variant: 'warning',
      });
      return leave;
    });

    return { selectedFile, consentAccepted, isDragging, uploading, uploadResult, finalStatus, error, progress, statusLines, phaseLabel, ocrLines, visibleOcrLines, ocrDone, checkingProfiles, processingProfile, handleDrop, handleFileSelect, doUpload, reset, goToProfile, goToProcessing };
  },
};

// ================================================================
// PAGES — Profile Detail
// ================================================================
const ProfileDetailPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div class="min-w-0">
            <div class="flex items-center gap-2 text-sm font-mono min-w-0">
              <router-link to="/profiles" class="font-bold hover:text-white transition flex-shrink-0" style="color:var(--text-dim)">PROFILES</router-link>
              <span style="color:var(--text-dim)" class="flex-shrink-0">/</span>
              <span class="font-bold text-white truncate">{{ info?.name || 'Profile' }}</span>
            </div>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Resume profile detail</p>
          </div>
        </template>
        <template #right>
          <div class="flex items-center gap-2 md:gap-3">
            <button v-if="editing" @click="cancelEdit" class="btn-ghost">CANCEL</button>
            <button v-if="editing" @click="saveEdit" class="btn-primary text-xs" :disabled="saving || store.loading">
              <span v-if="saving" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
              {{ saving ? 'SAVING...' : 'SAVE' }}
            </button>
            <button v-if="!editing && info && profile && profile.status === 'READY'" @click="startEdit" class="btn-ghost" :disabled="store.loading">EDIT</button>
            <button @click="goCreateJob" class="btn-ghost hidden md:inline-flex" v-if="!editing && profile && profile.status === 'READY'" :disabled="store.loading">CREATE JOB</button>
            <button v-if="!editing && profile" @click="doDelete" class="btn-ghost danger" :disabled="deleting || store.loading">
              <svg class="md:hidden" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
              <span class="hidden md:inline">DELETE</span>
            </button>
          </div>
        </template>
      </TopHeader>

      <!-- ====== READY + VIEW MODE: Split layout with cards + Chat ====== -->
      <div v-if="info && !editing && profile && profile.status === 'READY'" class="split-layout">
        <div class="split-main">
          <div class="max-w-3xl mx-auto space-y-6">
            <!-- CTA: Shown only when user has zero jobs (profile→job conversion nudge) -->
            <div v-if="showCta" class="widget-card fade-slide-in" style="border-left:3px solid var(--teal);background:rgba(1,169,219,0.06);">
              <div class="flex items-center justify-between gap-3 p-1">
                <div class="flex items-center gap-3 min-w-0">
                  <span class="text-lg flex-shrink-0">&#x1F3AF;</span>
                  <p class="text-xs text-white"><span class="font-semibold">Ready to tailor?</span> <span style="color:var(--text-dim)">Paste a job description and get a custom resume in seconds.</span></p>
                </div>
                <div class="flex items-center gap-3 flex-shrink-0">
                  <button @click="goCreateJob" class="btn-primary text-xs whitespace-nowrap">CREATE JOB</button>
                  <button @click="ctaDismissed = true" class="text-xs" style="color:var(--text-dim);cursor:pointer;background:none;border:none;padding:4px;">&#x2715;</button>
                </div>
              </div>
            </div>
            <!-- Error message -->
            <div v-if="editError" class="card-warning p-4">
              <p class="text-xs text-red-400 font-mono">{{ editError }}</p>
            </div>
            <div class="widget-card fade-slide-in">
              <div class="section-label">Personal Information</div>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div><label class="kpi-label block mb-1">Name</label><p class="text-white font-semibold">{{ info.name || '-' }}</p></div>
                <div><label class="kpi-label block mb-1">Email</label><p class="text-white">{{ info.email || '-' }}</p></div>
                <div><label class="kpi-label block mb-1">Phone</label><p class="text-white">{{ info.mobile_number || '-' }}</p></div>
                <div><label class="kpi-label block mb-1">Date of Birth</label><p class="text-white">{{ info.date_of_birth || '-' }}</p></div>
              </div>
            </div>
            <div v-if="info.links && info.links.length" class="widget-card fade-slide-in">
              <div class="section-label">Links</div>
              <div class="flex flex-wrap gap-3 mt-3">
                <a v-for="link in info.links" :key="link.url" :href="link.url" target="_blank"
                   class="badge cursor-pointer hover:opacity-80 transition" style="background:rgba(1,169,219,0.08);color:var(--teal);">
                  {{ link.name || link.url }}
                </a>
              </div>
            </div>
            <div v-if="info.skills && info.skills.length" class="widget-card fade-slide-in">
              <div class="section-label">Skills</div>
              <div class="space-y-3 mt-3">
                <div v-for="(skills, cat) in groupedSkills" :key="cat">
                  <p class="text-[10px] font-mono uppercase mb-1" style="color:var(--text-dim)">{{ cat }}</p>
                  <div class="flex flex-wrap gap-2">
                    <span v-for="skill in skills" :key="skill" class="badge" style="background:rgba(251,65,0,0.08);color:var(--orange);font-size:10px;">{{ skill }}</span>
                  </div>
                </div>
              </div>
            </div>
            <div v-if="info.past_experience && info.past_experience.length" class="widget-card fade-slide-in">
              <div class="section-label">Experience</div>
              <div class="space-y-4 mt-4">
                <div v-for="(exp, i) in info.past_experience" :key="i" class="p-4 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-1 mb-2">
                    <div>
                      <h4 class="text-white font-bold text-sm">{{ exp.role }}</h4>
                      <p class="text-xs text-slate-400">{{ [exp.company_name, exp.department].filter(Boolean).join(' — ') }}</p>
                      <p v-if="exp.location" class="text-[10px] text-slate-500">{{ exp.location }}</p>
                    </div>
                    <span class="font-mono text-[10px] text-slate-500 flex-shrink-0">{{ [exp.start_date, exp.end_date || 'Present'].filter(Boolean).join(' — ') }}</span>
                  </div>
                  <p v-if="exp.description" class="text-xs text-slate-400 mt-2 leading-relaxed whitespace-pre-line">{{ exp.description }}</p>
                </div>
              </div>
            </div>
            <div v-if="info.educations && info.educations.length" class="widget-card fade-slide-in">
              <div class="section-label">Education</div>
              <div class="space-y-3 mt-4">
                <div v-for="(edu, i) in info.educations" :key="i" class="p-3 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <h4 class="text-white font-semibold text-sm">{{ edu.degree }}</h4>
                  <p class="text-xs text-slate-400">{{ edu.institution }} {{ edu.grade ? '(' + edu.grade + ')' : '' }}</p>
                  <p v-if="edu.start_date || edu.end_date" class="font-mono text-[10px] text-slate-500 mt-1">{{ [edu.start_date, edu.end_date].filter(Boolean).join(' — ') }}</p>
                </div>
              </div>
            </div>
            <div v-if="info.projects && info.projects.length" class="widget-card fade-slide-in">
              <div class="section-label">Projects</div>
              <div class="space-y-3 mt-4">
                <div v-for="(proj, i) in info.projects" :key="i" class="p-3 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <h4 class="text-white font-semibold text-sm">{{ proj.name }}</h4>
                  <p class="text-xs text-slate-400 mt-1" v-if="proj.description">{{ proj.description }}</p>
                  <a v-if="proj.link" :href="proj.link" target="_blank" class="text-[10px] font-mono mt-1 inline-block" style="color:var(--teal);">{{ proj.link }}</a>
                </div>
              </div>
            </div>
            <div v-if="info.achievements && info.achievements.length" class="widget-card fade-slide-in">
              <div class="section-label">Achievements</div>
              <div class="space-y-3 mt-4">
                <div v-for="(ach, i) in info.achievements" :key="i" class="p-3 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <h4 class="text-white font-semibold text-sm">{{ ach.name }}</h4>
                  <p class="text-xs text-slate-400 mt-1" v-if="ach.description">{{ ach.description }}</p>
                </div>
              </div>
            </div>
            <div v-if="info.certifications && info.certifications.length" class="widget-card fade-slide-in">
              <div class="section-label">Certifications</div>
              <div class="space-y-2 mt-3">
                <div v-for="cert in info.certifications" :key="cert.name || cert" class="flex items-center gap-2">
                  <span class="badge" style="background:rgba(16,185,129,0.08);color:var(--green);font-size:10px;">{{ typeof cert === 'string' ? cert : cert.name }}</span>
                  <span v-if="cert.credential_id" class="font-mono text-[10px] text-slate-500">ID: {{ cert.credential_id }}</span>
                </div>
              </div>
            </div>
            <div v-if="info.patents && info.patents.length" class="widget-card fade-slide-in">
              <div class="section-label">Patents</div>
              <div class="space-y-3 mt-4">
                <div v-for="(pat, i) in info.patents" :key="i" class="p-3 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <h4 class="text-white font-semibold text-sm">{{ pat.name }}</h4>
                  <p class="text-xs text-slate-400 mt-1" v-if="pat.description">{{ pat.description }}</p>
                </div>
              </div>
            </div>
            <div v-if="info.papers && info.papers.length" class="widget-card fade-slide-in">
              <div class="section-label">Papers</div>
              <div class="space-y-3 mt-4">
                <div v-for="(paper, i) in info.papers" :key="i" class="p-3 rounded" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <h4 class="text-white font-semibold text-sm">{{ paper.name }}</h4>
                  <p class="text-xs text-slate-400 mt-1" v-if="paper.description">{{ paper.description }}</p>
                </div>
              </div>
            </div>
            <div class="h-8"></div>
          </div>
        </div>
        <ChatPanel entity-type="profile" :entity-id="profileId" :visible="chatOpen" @close="chatOpen = false" @modified="onChatModified" />
      </div>

      <!-- ====== NON-READY or EDITING: Standard scrollable layout ====== -->
      <div v-else class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div v-if="store.loading && !editing" class="flex items-center justify-center py-16"><div class="spinner"></div></div>
        <div v-else-if="!profile" class="empty-state py-16">
          <p class="text-lg text-white">Profile not found</p>
        </div>
        <div v-else class="max-w-3xl mx-auto">
          <!-- Processing state -->
          <div v-if="profile.status === 'PENDING' || profile.status === 'PROCESSING'" class="widget-card text-center py-12 mb-6">
            <div class="spinner mx-auto mb-4" style="width:32px;height:32px;border-width:3px;"></div>
            <p class="text-white font-semibold">Processing your resume...</p>
            <p class="text-xs" style="color:var(--text-dim)">This page will update automatically</p>
            <div class="mx-auto mt-4 px-4 py-3 rounded-lg text-left" style="background:rgba(251,65,0,0.08); border:1px solid rgba(251,65,0,0.2); max-width:320px;">
              <div class="flex items-center gap-2">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2" class="flex-shrink-0"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <p class="text-[11px]" style="color:var(--orange)">Usually 1-2 min, up to 3-4 min in rare cases.</p>
              </div>
            </div>
          </div>
          <!-- Failed state -->
          <div v-else-if="profile.status === 'FAILED'" class="card-warning p-6 mb-6">
            <h3 class="text-white font-bold text-sm font-mono mb-2">PROCESSING FAILED</h3>
            <p class="text-xs text-slate-400">The resume could not be processed. Please try again with a different file.</p>
          </div>
          <!-- Error message -->
          <div v-if="editError" class="card-warning p-4 mb-4">
            <p class="text-xs text-red-400 font-mono">{{ editError }}</p>
          </div>

          <!-- ====== EDIT MODE ====== -->
          <div v-if="editing && ed" class="space-y-6">
            <!-- Personal Info -->
            <div class="widget-card">
              <div class="section-label">Personal Information</div>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div><label class="kpi-label block mb-1">Name</label><input v-model="ed.name" class="input-field" /></div>
                <div><label class="kpi-label block mb-1">Email</label><input v-model="ed.email" class="input-field" /></div>
                <div><label class="kpi-label block mb-1">Phone</label><input v-model="ed.mobile_number" class="input-field" placeholder="e.g. +91 9876543210" /></div>
                <div><label class="kpi-label block mb-1">Date of Birth</label><input v-model="ed.date_of_birth" class="input-field" placeholder="YYYY-MM-DD" /></div>
              </div>
            </div>
            <!-- Links -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Links</div>
                <button @click="ed.links.push({name:'',url:''})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-2 mt-3">
                <div v-for="(link, i) in ed.links" :key="i" class="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                  <input v-model="link.name" class="input-field w-full md:w-[120px] md:flex-shrink-0" placeholder="e.g. github" />
                  <input v-model="link.url" class="input-field flex-1" placeholder="https://..." />
                  <button @click="ed.links.splice(i,1)" class="text-red-400 hover:text-red-300 text-xs flex-shrink-0 self-end md:self-center">X</button>
                </div>
              </div>
            </div>
            <!-- Skills -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Skills</div>
                <button @click="ed.skills.push({name:'',category:'Other'})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-2 mt-3">
                <div v-for="(skill, i) in ed.skills" :key="i" class="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                  <input v-model="skill.name" class="input-field flex-1" placeholder="Skill name" />
                  <input v-model="skill.category" class="input-field w-full md:w-[150px] md:flex-shrink-0" placeholder="Category" />
                  <button @click="ed.skills.splice(i,1)" class="text-red-400 hover:text-red-300 text-xs flex-shrink-0 self-end md:self-center">X</button>
                </div>
              </div>
            </div>
            <!-- Experience -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Experience</div>
                <button @click="ed.past_experience.push({company_name:'',department:null,location:null,role:'',start_date:null,end_date:null,description:''})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-4 mt-4">
                <div v-for="(exp, i) in ed.past_experience" :key="i" class="p-4 rounded relative" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <button @click="ed.past_experience.splice(i,1)" class="absolute top-2 right-2 text-red-400 hover:text-red-300 text-xs">X</button>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div><label class="kpi-label block mb-1">Role</label><input v-model="exp.role" class="input-field" /></div>
                    <div><label class="kpi-label block mb-1">Company</label><input v-model="exp.company_name" class="input-field" /></div>
                    <div><label class="kpi-label block mb-1">Department</label><input v-model="exp.department" class="input-field" placeholder="Optional" /></div>
                    <div><label class="kpi-label block mb-1">Location</label><input v-model="exp.location" class="input-field" placeholder="e.g. Remote" /></div>
                    <div><label class="kpi-label block mb-1">Start Date</label><input v-model="exp.start_date" class="input-field" placeholder="YYYY-MM" /></div>
                    <div><label class="kpi-label block mb-1">End Date</label><input v-model="exp.end_date" class="input-field" placeholder="YYYY-MM or leave empty" /></div>
                  </div>
                  <div class="mt-3"><label class="kpi-label block mb-1">Description</label><textarea v-model="exp.description" class="input-field" rows="4"></textarea></div>
                </div>
              </div>
            </div>
            <!-- Education -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Education</div>
                <button @click="ed.educations.push({degree:'',institution:'',grade:null,start_date:null,end_date:null})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-3 mt-4">
                <div v-for="(edu, i) in ed.educations" :key="i" class="p-3 rounded relative" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <button @click="ed.educations.splice(i,1)" class="absolute top-2 right-2 text-red-400 hover:text-red-300 text-xs">X</button>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div><label class="kpi-label block mb-1">Degree</label><input v-model="edu.degree" class="input-field" /></div>
                    <div><label class="kpi-label block mb-1">Institution</label><input v-model="edu.institution" class="input-field" /></div>
                    <div><label class="kpi-label block mb-1">Grade</label><input v-model="edu.grade" class="input-field" placeholder="e.g. 8.72 CGPA" /></div>
                    <div class="grid grid-cols-2 gap-2">
                      <div><label class="kpi-label block mb-1">Start</label><input v-model="edu.start_date" class="input-field" placeholder="YYYY" /></div>
                      <div><label class="kpi-label block mb-1">End</label><input v-model="edu.end_date" class="input-field" placeholder="YYYY" /></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <!-- Projects -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Projects</div>
                <button @click="ed.projects.push({name:'',description:'',link:null})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-3 mt-4">
                <div v-for="(proj, i) in ed.projects" :key="i" class="p-3 rounded relative" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
                  <button @click="ed.projects.splice(i,1)" class="absolute top-2 right-2 text-red-400 hover:text-red-300 text-xs">X</button>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div><label class="kpi-label block mb-1">Name</label><input v-model="proj.name" class="input-field" /></div>
                    <div><label class="kpi-label block mb-1">Link</label><input v-model="proj.link" class="input-field" placeholder="Optional URL" /></div>
                  </div>
                  <div class="mt-3"><label class="kpi-label block mb-1">Description</label><textarea v-model="proj.description" class="input-field" rows="3"></textarea></div>
                </div>
              </div>
            </div>
            <!-- Achievements -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Achievements</div>
                <button @click="ed.achievements.push({name:'',description:null})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-2 mt-3">
                <div v-for="(ach, i) in ed.achievements" :key="i" class="flex items-start gap-2">
                  <div class="flex-1 space-y-2">
                    <input v-model="ach.name" class="input-field" placeholder="Achievement name" />
                    <input v-model="ach.description" class="input-field" placeholder="Description (optional)" />
                  </div>
                  <button @click="ed.achievements.splice(i,1)" class="text-red-400 hover:text-red-300 text-xs mt-2">X</button>
                </div>
              </div>
            </div>
            <!-- Certifications -->
            <div class="widget-card">
              <div class="flex items-center justify-between">
                <div class="section-label">Certifications</div>
                <button @click="ed.certifications.push({name:'',credential_id:null})" class="btn-ghost text-[10px]">+ ADD</button>
              </div>
              <div class="space-y-2 mt-3">
                <div v-for="(cert, i) in ed.certifications" :key="i" class="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                  <input v-model="cert.name" class="input-field flex-1" placeholder="Certification name" />
                  <input v-model="cert.credential_id" class="input-field w-full md:w-[160px] md:flex-shrink-0" placeholder="Credential ID" />
                  <button @click="ed.certifications.splice(i,1)" class="text-red-400 hover:text-red-300 text-xs flex-shrink-0 self-end md:self-center">X</button>
                </div>
              </div>
            </div>
          </div>
          <div class="h-8"></div>
        </div>
      </div>

      <!-- Mobile: chat backdrop + FAB -->
      <div v-if="showChatFab" class="chat-backdrop" :class="{ visible: chatOpen }" @click="chatOpen = false"></div>
      <button v-if="showChatFab && !chatOpen" class="chat-fab" @click="chatOpen = true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      </button>
    </div>
  `,
  setup() {
    const router = useRouter();
    const store = useProfileStore();
    const jobStore = useJobStore();
    const editing = ref(false);
    const saving = ref(false);
    const editError = ref(null);
    const ed = ref(null);
    const chatOpen = ref(false);
    const ctaDismissed = ref(false);
    const profileId = computed(() => store.selectedId);
    const profile = computed(() => store.current);
    const info = computed(() => profile.value?.resume_info || null);

    // Show CTA only for users with zero jobs (the conversion gap we're targeting)
    const showCta = computed(() =>
      profile.value?.status === 'READY' && !editing.value && jobStore.total === 0 && !ctaDismissed.value
    );

    // Show chat FAB only when READY, has info, and not editing
    const showChatFab = computed(() => {
      return profile.value?.status === 'READY' && !!info.value && !editing.value;
    });

    // ATS Readiness checklist — purely computed from resume_info, no API needed
    const atsChecks = computed(() => {
      const d = info.value;
      if (!d) return [];

      // --- gather data ---
      const links = d.links || [];
      const skills = d.skills || [];
      const exps = d.past_experience || [];
      const edus = d.educations || [];
      const projects = d.projects || [];

      // Contact
      const hasEmail = !!d.email;
      const hasPhone = !!d.mobile_number;

      // Links detection
      const hasLinkedIn = links.some(l => /linkedin/i.test(l.url || '') || /linkedin/i.test(l.name || ''));
      const hasGithub = links.some(l => /github/i.test(l.url || '') || /github/i.test(l.name || ''));
      const hasPortfolio = links.some(l => /portfolio|website|personal/i.test(l.name || '') || /\.dev|\.me|\.io/i.test(l.url || ''));
      const foundLinks = [hasLinkedIn && 'LinkedIn', hasGithub && 'GitHub', hasPortfolio && 'Portfolio'].filter(Boolean);

      // Skills analysis
      const skillCategories = new Set(skills.map(s => (typeof s === 'string') ? 'Other' : (s.category || 'Other')));
      const hasEnoughSkills = skills.length >= 5;
      const hasDiverseSkills = skillCategories.size >= 2 && !([...skillCategories].length === 1 && skillCategories.has('Other'));

      // Experience analysis
      const allDescriptions = [...exps.map(e => e.description || ''), ...projects.map(p => p.description || '')].join(' ');
      const quantifiedPattern = /\d+\s*%|\d+x\b|\$[\d,.]+[KkMmBb]?|\b\d{2,}[+]?\s*(users|customers|clients|requests|transactions|servers|endpoints|apis|records|entries)/i;
      const hasQuantified = quantifiedPattern.test(allDescriptions);
      const actionVerbs = /^(Led|Built|Designed|Developed|Implemented|Architected|Launched|Reduced|Increased|Improved|Optimized|Automated|Delivered|Managed|Created|Engineered|Scaled|Deployed|Migrated|Integrated|Streamlined|Spearheaded|Orchestrated|Resolved|Revamped|Established)/m;
      const descWithVerbs = exps.filter(e => actionVerbs.test((e.description || '').trim()));
      const hasActionVerbs = exps.length === 0 || descWithVerbs.length >= Math.ceil(exps.length * 0.5);

      // Experience description depth — at least 50 chars per role
      const thinDescriptions = exps.filter(e => (e.description || '').trim().length < 50);

      // Date completeness — experience and education should have dates for ATS tenure parsing
      const expMissingDates = exps.filter(e => !e.start_date);
      const eduMissingDates = edus.filter(e => !e.start_date && !e.end_date);

      // --- build checklist ---
      const checks = [];

      // 1. Format (always pass — LaTeX)
      checks.push({
        label: 'ATS-Parseable Format',
        pass: true,
        detail: 'Compiled with LaTeX using standard section headers (Experience, Education, Skills, Projects). ATS parsers can read every field cleanly — no columns, tables, or image-based layouts that break parsing.',
      });

      // 2. Contact info
      checks.push({
        label: 'Contact Information',
        pass: hasEmail && hasPhone,
        detail: hasEmail && hasPhone
          ? 'Email and phone number present. ATS extracts these first to create your candidate profile.'
          : 'Missing ' + [!hasEmail && 'email', !hasPhone && 'phone number'].filter(Boolean).join(' and ') + '. Most ATS systems reject profiles without complete contact info.',
      });

      // 3. Profile links
      checks.push({
        label: 'Profile Links',
        pass: foundLinks.length > 0,
        detail: foundLinks.length > 0
          ? 'Found: ' + foundLinks.join(', ') + '. ' + (foundLinks.length < 2 ? 'Adding ' + ['LinkedIn', 'GitHub', 'Portfolio'].filter(l => !foundLinks.includes(l)).slice(0, 1).join('') + ' would strengthen this further.' : 'Recruiters cross-reference these to verify your work.')
          : 'No LinkedIn, GitHub, or portfolio link detected. 72% of recruiters check LinkedIn before shortlisting — add at least one link.',
      });

      // 4. Skills coverage
      checks.push({
        label: 'Skills & Keywords',
        pass: hasEnoughSkills && hasDiverseSkills,
        detail: (() => {
          if (hasEnoughSkills && hasDiverseSkills) return skills.length + ' skills across ' + skillCategories.size + ' categories. Well-structured for ATS keyword matching against job descriptions.';
          const issues = [];
          if (!hasEnoughSkills) issues.push('only ' + skills.length + ' skill(s) listed — aim for 8-15');
          if (!hasDiverseSkills) issues.push('skills lack categorization — group into Programming, Frameworks, Tools, etc.');
          return issues.join('; ') + '. ATS systems match JD keywords against this section — it\'s the single most important section for keyword hits.';
        })(),
      });

      // 5. Experience presence + description depth
      checks.push({
        label: 'Experience Depth',
        pass: exps.length > 0 && thinDescriptions.length === 0,
        detail: (() => {
          if (exps.length === 0) return 'No work experience listed. Add internships, freelance, research, or TA roles — ATS filters heavily on experience.';
          if (thinDescriptions.length > 0) return thinDescriptions.length + ' of ' + exps.length + ' role(s) have thin descriptions (< 50 chars). ATS systems extract keywords from descriptions — flesh them out with specific technologies, scope, and outcomes.';
          return exps.length + ' role(s) with substantive descriptions. ATS can extract relevant keywords and context from each.';
        })(),
      });

      // 6. Quantified impact (across experience + projects)
      checks.push({
        label: 'Quantified Achievements',
        pass: hasQuantified,
        detail: hasQuantified
          ? 'Found measurable results (numbers, percentages, scale) in your descriptions. Quantified impact makes you stand out in both ATS ranking and human review.'
          : 'No quantified achievements detected. Add metrics like "reduced load time by 40%", "served 10K+ users", or "managed $2M budget". Resumes with numbers get 2x more interviews.',
      });

      // 7. Action verbs
      checks.push({
        label: 'Action-Oriented Language',
        pass: hasActionVerbs,
        detail: hasActionVerbs
          ? 'Descriptions use strong action verbs (Led, Built, Optimized, etc.). This signals ownership and impact to both ATS parsers and hiring managers.'
          : 'Some role descriptions don\'t start with action verbs. Rewrite with verbs like Built, Led, Designed, Reduced, Implemented — ATS systems and recruiters both respond better to active voice.',
      });

      // 8. Date consistency
      const totalMissing = expMissingDates.length + eduMissingDates.length;
      checks.push({
        label: 'Timeline & Dates',
        pass: totalMissing === 0,
        detail: totalMissing === 0
          ? 'All experience and education entries have dates. ATS uses these to calculate tenure, detect gaps, and verify minimum experience requirements.'
          : totalMissing + ' entry/entries missing dates. ATS systems use dates to calculate years of experience and filter by "minimum X years required" — missing dates often means auto-rejection.',
      });

      // 9. Education
      checks.push({
        label: 'Education',
        pass: edus.length > 0,
        detail: edus.length > 0
          ? edus.length + ' education entry/entries with degree and institution. ATS can parse and match against degree requirements.'
          : 'No education listed. Many ATS systems hard-filter by degree — even self-taught candidates should list relevant coursework or certifications.',
      });

      // 10. Projects (especially valuable for students / < 2 exp)
      if (exps.length < 2) {
        checks.push({
          label: 'Projects',
          pass: projects.length >= 2,
          detail: projects.length >= 2
            ? projects.length + ' projects listed. For early-career candidates, projects are your primary evidence of technical ability — ATS picks up technologies and keywords from here.'
            : (projects.length === 0 ? 'No' : 'Only ' + projects.length) + ' project(s) listed. With limited work experience, aim for 2-4 projects with descriptions mentioning specific technologies, scale, and outcomes.',
        });
      }

      return checks;
    });

    const groupedSkills = computed(() => {
      if (!info.value?.skills) return {};
      const groups = {};
      for (const s of info.value.skills) {
        const cat = (typeof s === 'string') ? 'Other' : (s.category || s.type || 'Other');
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(typeof s === 'string' ? s : s.name);
      }
      return groups;
    });

    // Chat modification handler — update the store so left panel re-renders live
    const onChatModified = (data) => {
      if (data.resume_info && store.current) {
        store.current.resume_info = data.resume_info;
      }
    };

    const startEdit = () => {
      chatOpen.value = false; // close chat drawer on mobile when editing
      ed.value = JSON.parse(JSON.stringify(info.value));
      ed.value.links = ed.value.links || [];
      ed.value.skills = ed.value.skills || [];
      ed.value.past_experience = ed.value.past_experience || [];
      ed.value.educations = ed.value.educations || [];
      ed.value.projects = ed.value.projects || [];
      ed.value.achievements = ed.value.achievements || [];
      ed.value.certifications = ed.value.certifications || [];
      ed.value.patents = ed.value.patents || [];
      ed.value.papers = ed.value.papers || [];
      editing.value = true;
      editError.value = null;
    };
    const cancelEdit = () => { editing.value = false; ed.value = null; editError.value = null; };
    const saveEdit = async () => {
      saving.value = true;
      editError.value = null;
      try {
        await store.updateProfile(profileId.value, ed.value);
        editing.value = false;
        ed.value = null;
      } catch (e) {
        editError.value = e.message;
      } finally { saving.value = false; }
    };
    const goCreateJob = () => {
      router.push('/jobs/new');
    };
    const deleting = ref(false);
    const doDelete = async () => {
      const ok = await showConfirm({ title: 'Delete Profile', message: 'This profile and all associated data will be permanently removed. This cannot be undone.' });
      if (!ok) return;
      deleting.value = true;
      try {
        await store.deleteProfile(profileId.value);
        router.replace('/profiles');
      } catch (e) {
        editError.value = e.message;
      } finally { deleting.value = false; }
    };

    let unmounted = false;
    onMounted(async () => {
      if (!profileId.value) { router.replace('/profiles'); return; }
      // Fetch profile + jobs in parallel (jobs needed for CTA visibility)
      await Promise.all([
        store.fetchProfile(profileId.value),
        jobStore.fetchJobs(),
      ]);
      if (profile.value && (profile.value.status === 'PENDING' || profile.value.status === 'PROCESSING')) {
        store.pollStatus(profileId.value).then(() => {
          if (!unmounted) store.fetchProfile(profileId.value);
        });
      }
    });
    onUnmounted(() => { unmounted = true; });
    return { store, profile, info, profileId, groupedSkills, atsChecks, chatOpen, showChatFab, showCta, ctaDismissed, editing, saving, deleting, editError, ed, onChatModified, startEdit, cancelEdit, saveEdit, goCreateJob, doDelete };
  },
};

// ================================================================
// PAGES — Job List
// ================================================================
const JobListPage = {
  components: { StatusBadge, PaginationControls },
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <h1 class="text-sm font-bold text-white font-mono tracking-tight">JOBS</h1>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Resume generation jobs</p>
          </div>
        </template>
        <template #right>
          <router-link to="/jobs/new" class="btn-ghost">+ NEW JOB</router-link>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div v-if="store.loading && store.jobs.length === 0" class="flex items-center justify-center py-16"><div class="spinner"></div></div>
        <div v-else-if="store.jobs.length === 0" class="empty-state py-16">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          <p class="text-lg font-semibold text-white mt-2">No jobs yet</p>
          <p class="text-sm mb-4">Create a job to generate a tailored resume</p>
          <router-link to="/jobs/new" class="btn-primary">CREATE JOB</router-link>
        </div>
        <template v-else>
          <div class="space-y-3">
            <div v-for="j in store.jobs" :key="j.id"
                 class="widget-card fade-slide-in">
              <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
                <div class="flex-1 min-w-0 cursor-pointer" @click="goJob(j.id)">
                  <div class="flex items-center gap-3 mb-1">
                    <h3 class="text-sm text-white font-bold truncate">{{ jobLabel(j) }}</h3>
                    <StatusBadge :status="j.status" />
                  </div>
                  <p class="text-xs text-slate-400 truncate" v-if="j.job_description && j.job_description.description">
                    {{ j.job_description.description.length > 100 ? j.job_description.description.slice(0, 100) + '...' : j.job_description.description }}
                  </p>
                </div>
                <div class="flex items-center gap-3 flex-shrink-0">
                  <span class="font-mono text-[10px] text-slate-600">{{ formatDate(j.created_at) }}</span>
                  <button v-if="j.status === 'READY'"
                          @click.stop="doDownload(j)"
                          class="btn-ghost text-xs flex items-center gap-1"
                          :disabled="downloadingId === j.id">
                    <svg v-if="downloadingId !== j.id" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                    <span v-if="downloadingId === j.id" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
                    PDF
                  </button>
                </div>
              </div>
            </div>
          </div>
          <PaginationControls :page="store.page" :totalPages="store.totalPages" @go="store.goToPage" />
        </template>
      </div>
    </div>
  `,
  setup() {
    const router = useRouter();
    const store = useJobStore();
    const downloadingId = ref(null);
    let refreshTimer = null;
    const activeStatuses = ['GENERATING_RESUME', 'RESUME_GENERATED', 'GENERATING_PDF'];
    const hasActiveJobs = () => store.jobs.some(j => activeStatuses.includes(j.status));

    onMounted(() => {
      store.fetchJobs(true, { limit: 10 });
      refreshTimer = setInterval(() => {
        if (hasActiveJobs()) store.fetchJobs(true);
      }, 5000);
    });
    onUnmounted(() => { if (refreshTimer) clearInterval(refreshTimer); });

    const goJob = (id) => { store.selectJob(id); router.push('/jobs/view'); };
    const doDownload = async (j) => {
      downloadingId.value = j.id;
      try {
        const jd = j.job_description || {};
        await downloadJobPdf(j.id, j.candidate_name, jd.role, jd.company);
      } catch (e) {
        alert('Download failed: ' + e.message);
      } finally {
        downloadingId.value = null;
      }
    };
    return { store, downloadingId, formatDate, jobLabel, goJob, doDownload };
  },
};

// ================================================================
// PAGES — Job Create
// ================================================================
const JobCreatePage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <div class="flex items-center gap-2 text-sm font-mono">
              <router-link to="/jobs" class="font-bold hover:text-white transition" style="color:var(--text-dim)">JOBS</router-link>
              <span style="color:var(--text-dim)">/</span>
              <span class="font-bold text-white">New</span>
            </div>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Generate a tailored resume for a job description</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div class="max-w-2xl mx-auto">
          <!-- ====== FORM (visible when not processing) ====== -->
          <template v-if="!processing && !finalStatus">
            <!-- Step 1: Select Profile -->
            <div class="widget-card mb-6 fade-slide-in stagger-1">
              <div class="section-label">Step 1: Select Profile</div>
              <div v-if="profileStore.loading" class="flex items-center gap-2 mt-4">
                <div class="spinner"></div><span class="text-xs text-slate-400">Loading profiles...</span>
              </div>
              <div v-else-if="readyProfiles.length === 0" class="mt-4">
                <p class="text-sm text-slate-400 mb-3">No ready profiles available.</p>
                <router-link to="/profiles/new" class="btn-secondary text-xs">Create a Profile First</router-link>
              </div>
              <div v-else class="mt-4 space-y-2" style="max-height:200px;overflow-y:auto;">
                <div v-for="p in readyProfiles" :key="p.id"
                     class="p-3 rounded cursor-pointer transition-all"
                     :style="{
                       background: selectedProfile === p.id ? 'rgba(251,65,0,0.08)' : 'rgba(0,0,0,0.15)',
                       border: selectedProfile === p.id ? '1px solid var(--orange)' : '1px solid rgba(255,255,255,0.04)',
                       boxShadow: selectedProfile === p.id ? '0 0 20px rgba(251,65,0,0.1)' : 'none'
                     }"
                     @click="selectedProfile = p.id">
                  <div class="flex items-center justify-between">
                    <div>
                      <span class="text-white font-semibold text-sm">{{ p.resume_info?.name || 'Untitled Profile' }}</span>
                      <span class="text-xs text-slate-500 ml-2">{{ p.resume_info?.email || '' }}</span>
                    </div>
                    <div class="flex items-center gap-3">
                      <span class="font-mono text-[10px] text-slate-600">{{ formatDate(p.created_at, { includeYear: true }) }}</span>
                      <div v-if="selectedProfile === p.id" style="color:var(--orange);">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 8L7 11L12 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <!-- Step 2: Job Details -->
            <div class="widget-card mb-6 fade-slide-in stagger-2">
              <div class="section-label">Step 2: Job Details</div>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div>
                  <label class="kpi-label block mb-1">Company</label>
                  <input v-model="company" class="input-field" placeholder="e.g. Google" />
                </div>
                <div>
                  <label class="kpi-label block mb-1">Role</label>
                  <input v-model="role" class="input-field" placeholder="e.g. Software Engineer" />
                </div>
              </div>
              <label class="kpi-label block mb-1 mt-4">Job Description</label>
              <textarea v-model="jobDescription"
                        class="input-field"
                        placeholder="Paste the full job description here...&#10;&#10;Include requirements, responsibilities, and qualifications."
                        rows="10"></textarea>
              <p class="text-[10px] font-mono mt-2" style="color:var(--text-dim)">
                {{ jobDescription.length }} characters
              </p>
            </div>
            <!-- Submit -->
            <div class="fade-slide-in stagger-3">
              <button class="btn-primary w-full justify-center" @click="submit"
                      :disabled="!selectedProfile || !company.trim() || !role.trim() || !jobDescription.trim() || submitting || profileStore.loading">
                <span v-if="submitting" class="spinner" style="width:16px;height:16px;border-width:2px;"></span>
                {{ submitting ? 'CREATING...' : 'GENERATE RESUME' }}
              </button>
              <p v-if="error" class="text-red-400 text-xs mt-3 text-center font-mono">{{ error }}</p>
            </div>
          </template>

          <!-- ====== PROCESSING STATE ====== -->
          <template v-if="processing">
            <!-- Job info badge -->
            <div class="widget-card mb-4 fade-slide-in">
              <div class="flex items-center gap-3">
                <div class="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center" style="background:rgba(251,65,0,0.1);">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="1.5"><path d="M21 13V6a2 2 0 00-2-2H5a2 2 0 00-2 2v14l4-4h7"/><path d="M15 18l2 2 4-4"/></svg>
                </div>
                <div class="min-w-0">
                  <p class="text-white font-semibold text-sm truncate">{{ role }} at {{ company }}</p>
                  <p class="text-[10px] font-mono" style="color:var(--text-dim)">Generating tailored resume...</p>
                </div>
              </div>
            </div>

            <!-- Status + progress -->
            <div class="widget-card mb-4 fade-slide-in">
              <div class="flex items-center gap-3 mb-3">
                <div class="spinner" style="width:20px;height:20px;border-width:2px;"></div>
                <p class="text-white font-semibold text-sm font-mono">{{ phaseLabel }}</p>
              </div>
              <div class="progress-track" style="max-width:100%;">
                <div class="progress-fill transition-all duration-700" :style="{ width: progress + '%' }"></div>
              </div>
              <div class="mt-3 space-y-1">
                <div v-for="(line, i) in statusLines" :key="i" class="boot-line visible" :class="{ ok: line.ok }">
                  {{ line.text }}
                </div>
              </div>
            </div>

            <!-- JD text cycling — "reading job description" -->
            <div v-if="jdLines.length > 0 && !jdDone" class="widget-card fade-slide-in" style="border-left: 2px solid var(--teal);">
              <div class="flex items-center gap-2 mb-3">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                <span class="text-[10px] font-mono font-bold tracking-widest" style="color:var(--teal)">ANALYZING JOB DESCRIPTION</span>
              </div>
              <div class="overflow-hidden relative flex flex-col justify-end" style="height:180px;">
                <div class="space-y-0.5">
                  <div v-for="(line, i) in visibleJdLines" :key="i"
                       class="font-mono text-[11px] leading-relaxed transition-opacity duration-300"
                       :style="{ color: i === visibleJdLines.length - 1 ? 'var(--text-primary)' : 'var(--text-dim)', opacity: i < visibleJdLines.length - 3 ? 0.3 : (i < visibleJdLines.length - 1 ? 0.6 : 1) }">
                    <span v-if="i === visibleJdLines.length - 1" style="color:var(--teal);">&gt; </span>{{ line }}
                  </div>
                </div>
              </div>
            </div>

            <!-- AI tailoring — shown after JD cycling finishes while AI still works -->
            <div v-if="jdDone" class="widget-card fade-slide-in" style="border-left: 2px solid var(--orange);">
              <div class="flex items-center gap-3">
                <div class="spinner" style="width:18px;height:18px;border-width:2px;border-color:rgba(251,65,0,0.2);border-top-color:var(--orange);"></div>
                <div>
                  <p class="text-sm font-mono font-semibold" style="color:var(--orange)">TAILORING WITH AI</p>
                  <p class="text-[10px] font-mono mt-0.5" style="color:var(--text-dim)">Analysis complete — crafting your tailored resume...</p>
                </div>
              </div>
              <div class="mt-3 px-3 py-2 rounded" style="background:rgba(251,65,0,0.06); border:1px solid rgba(251,65,0,0.15);">
                <div class="flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2" class="flex-shrink-0"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  <p class="text-[11px] font-mono" style="color:var(--orange)">Usually 1-2 min, up to 3-4 min in rare cases. Please stay on this page.</p>
                </div>
              </div>
            </div>
          </template>

          <!-- ====== DONE (failure only — success auto-navigates) ====== -->
          <div v-if="finalStatus && !processing" class="widget-card text-center py-12 fade-slide-in">
            <div class="mb-4">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5" class="mx-auto">
                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <p class="text-white font-semibold mb-1">Generation failed</p>
            <p class="text-xs mb-6" style="color:var(--text-dim)">Something went wrong. You can retry or view the job details.</p>
            <div class="flex items-center justify-center gap-3">
              <button @click="retryGeneration" class="btn-primary" :disabled="submitting">
                <span v-if="submitting" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
                RETRY
              </button>
              <button @click="goToJob" class="btn-secondary">VIEW JOB</button>
              <button @click="reset" class="btn-secondary">NEW JOB</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const router = useRouter();
    const profileStore = useProfileStore();
    const jobStore = useJobStore();
    const creditStore = useCreditStore();
    const selectedProfile = ref(null);
    const company = ref('');
    const role = ref('');
    const jobDescription = ref('');
    const submitting = ref(false);
    const error = ref(null);
    let unmounted = false;

    // Processing state
    const processing = ref(false);
    const finalStatus = ref(null);
    const createdJobId = ref(null);
    const progress = ref(0);
    const statusLines = ref([]);
    const phaseLabel = ref('Creating job...');

    // JD text cycling
    const jdLines = ref([]);
    const visibleJdLines = ref([]);
    const jdDone = ref(false);
    let jdTimer = null;
    let progressTimerRef = null;
    const MAX_VISIBLE_LINES = 9;

    const readyProfiles = computed(() => profileStore.profiles.filter(p => p.status === 'READY'));

    onMounted(async () => {
      await profileStore.fetchProfiles(true, { limit: 50 });
      if (profileStore.selectedId) selectedProfile.value = profileStore.selectedId;
    });

    const startJdCycling = (companyName, roleName, description) => {
      // Build lines: company header, role header, blank, then JD lines
      const lines = [];
      lines.push('COMPANY: ' + companyName);
      lines.push('ROLE: ' + roleName);
      lines.push('---');
      const descLines = description.split(/\n/).map(l => l.trim()).filter(l => l.length > 0);
      lines.push(...descLines);
      if (lines.length === 0) return;
      jdLines.value = lines;
      jdDone.value = false;
      let idx = 0;
      visibleJdLines.value = [lines[0]];
      idx = 1;
      const delay = Math.max(150, Math.min(350, 20000 / lines.length));
      jdTimer = setInterval(() => {
        if (idx < lines.length) {
          const next = [...visibleJdLines.value, lines[idx]];
          visibleJdLines.value = next.length > MAX_VISIBLE_LINES ? next.slice(-MAX_VISIBLE_LINES) : next;
          idx++;
        } else {
          clearInterval(jdTimer);
          jdTimer = null;
          jdDone.value = true;
        }
      }, delay);
    };

    const stopCycling = () => {
      if (jdTimer) { clearInterval(jdTimer); jdTimer = null; }
      if (progressTimerRef) { clearInterval(progressTimerRef); progressTimerRef = null; }
    };

    const submit = async () => {
      if (!selectedProfile.value || !company.value.trim() || !role.value.trim() || !jobDescription.value.trim()) return;
      submitting.value = true;
      error.value = null;

      try {
        // Step 1: Create the job
        const jd = { company: company.value.trim(), role: role.value.trim(), description: jobDescription.value.trim() };
        const result = await jobStore.createJob(selectedProfile.value, jd);
        createdJobId.value = result.job_id;

        // Switch to processing view
        processing.value = true;
        submitting.value = false;
        progress.value = 10;
        phaseLabel.value = 'Submitting to AI...';
        statusLines.value = [{ text: '[SYS] Job created', ok: true }];

        // Step 2: Trigger generation (handles 429 paywall)
        try {
          await jobStore.generateResume(result.job_id);
          if (unmounted) return;
          creditStore.fetchBalance();
        } catch (genErr) {
          if (unmounted) return;
          if (genErr.is429) {
            phaseLabel.value = 'Waiting for credits...';
            statusLines.value.push({ text: '[CREDITS] Insufficient credits', ok: false });
            const bought = await showPaywall();
            if (unmounted) return;
            if (bought) {
              statusLines.value.push({ text: '[CREDITS] Payment received', ok: true });
              try {
                await jobStore.generateResume(result.job_id);
                if (unmounted) return;
                creditStore.fetchBalance();
              } catch (retryErr) {
                if (unmounted) return;
                // Generation failed after purchase — go to failed state
                statusLines.value.push({ text: '[SYS] Generation failed', ok: false });
                stopCycling();
                processing.value = false;
                finalStatus.value = 'FAILED';
                return;
              }
            } else {
              // User dismissed paywall — go to job detail
              stopCycling();
              processing.value = false;
              jobStore.selectJob(result.job_id);
              router.push('/jobs/view');
              return;
            }
          } else {
            // Non-429 error
            statusLines.value.push({ text: '[SYS] ' + genErr.message, ok: false });
            stopCycling();
            processing.value = false;
            finalStatus.value = 'FAILED';
            return;
          }
        }

        // Step 3: Generation triggered — start cycling + polling
        progress.value = 25;
        phaseLabel.value = 'Analyzing job description...';
        statusLines.value.push({ text: '[AI] Generation started', ok: true });
        statusLines.value.push({ text: '[AI] Tailoring resume to job description...', ok: false });

        startJdCycling(company.value.trim(), role.value.trim(), jobDescription.value.trim());

        // Smooth progress while polling
        progressTimerRef = setInterval(() => {
          if (progress.value < 85) progress.value += 2;
        }, 1500);

        // Poll for completion
        const status = await jobStore.pollStatus(result.job_id);
        stopCycling();
        if (unmounted) return;

        // Show last window of JD lines
        if (jdLines.value.length > 0) {
          visibleJdLines.value = jdLines.value.slice(-MAX_VISIBLE_LINES);
        }

        progress.value = 100;

        if (status === 'READY') {
          phaseLabel.value = 'Resume ready!';
          statusLines.value.push({ text: '[SYS] Tailored resume generated!', ok: true });
          statusLines.value.push({ text: '[TEX] PDF compiled successfully!', ok: true });
          await new Promise(r => setTimeout(r, 600));
          if (unmounted) return;
          // Navigate to job detail
          jobStore.selectJob(result.job_id);
          jobStore.fetchJobs(true);
          processing.value = false;
          router.push('/jobs/view');
          return;
        }

        // Failed
        phaseLabel.value = 'Generation failed';
        statusLines.value.push({ text: '[SYS] Generation failed', ok: false });
        await new Promise(r => setTimeout(r, 500));
        processing.value = false;
        finalStatus.value = 'FAILED';
      } catch (e) {
        stopCycling();
        if (unmounted) return;
        error.value = e.message;
        processing.value = false;
        submitting.value = false;
      }
    };

    const retryGeneration = async () => {
      if (!createdJobId.value) return;
      submitting.value = true;
      finalStatus.value = null;
      processing.value = true;
      progress.value = 15;
      phaseLabel.value = 'Retrying generation...';
      statusLines.value = [{ text: '[SYS] Retrying...', ok: false }];
      jdLines.value = [];
      visibleJdLines.value = [];
      jdDone.value = false;

      try {
        await jobStore.generateResume(createdJobId.value);
        if (unmounted) return;
        creditStore.fetchBalance();
      } catch (genErr) {
        if (unmounted) return;
        if (genErr.is429) {
          const bought = await showPaywall();
          if (unmounted) return;
          if (bought) {
            try {
              await jobStore.generateResume(createdJobId.value);
              if (unmounted) return;
              creditStore.fetchBalance();
            } catch (retryErr) {
              if (unmounted) return;
              stopCycling();
              processing.value = false;
              finalStatus.value = 'FAILED';
              submitting.value = false;
              return;
            }
          } else {
            stopCycling();
            processing.value = false;
            jobStore.selectJob(createdJobId.value);
            router.push('/jobs/view');
            submitting.value = false;
            return;
          }
        } else {
          stopCycling();
          processing.value = false;
          finalStatus.value = 'FAILED';
          submitting.value = false;
          return;
        }
      }

      submitting.value = false;
      progress.value = 25;
      statusLines.value.push({ text: '[AI] Generation started', ok: true });
      statusLines.value.push({ text: '[AI] Tailoring resume to job description...', ok: false });

      startJdCycling(company.value.trim(), role.value.trim(), jobDescription.value.trim());
      progressTimerRef = setInterval(() => {
        if (progress.value < 85) progress.value += 2;
      }, 1500);

      const status = await jobStore.pollStatus(createdJobId.value);
      stopCycling();
      if (unmounted) return;
      if (jdLines.value.length > 0) visibleJdLines.value = jdLines.value.slice(-MAX_VISIBLE_LINES);
      progress.value = 100;

      if (status === 'READY') {
        phaseLabel.value = 'Resume ready!';
        statusLines.value.push({ text: '[SYS] Tailored resume generated!', ok: true });
        await new Promise(r => setTimeout(r, 600));
        if (unmounted) return;
        jobStore.selectJob(createdJobId.value);
        jobStore.fetchJobs(true);
        processing.value = false;
        router.push('/jobs/view');
        return;
      }

      phaseLabel.value = 'Generation failed';
      statusLines.value.push({ text: '[SYS] Generation failed', ok: false });
      await new Promise(r => setTimeout(r, 500));
      processing.value = false;
      finalStatus.value = 'FAILED';
    };

    const goToJob = () => {
      if (createdJobId.value) {
        jobStore.selectJob(createdJobId.value);
        router.push('/jobs/view');
      }
    };

    const reset = () => {
      selectedProfile.value = null;
      company.value = '';
      role.value = '';
      jobDescription.value = '';
      submitting.value = false;
      error.value = null;
      processing.value = false;
      finalStatus.value = null;
      createdJobId.value = null;
      progress.value = 0;
      statusLines.value = [];
      jdLines.value = [];
      visibleJdLines.value = [];
      jdDone.value = false;
      stopCycling();
    };

    onUnmounted(() => { unmounted = true; stopCycling(); });

    onBeforeRouteLeave(async () => {
      if (!processing.value) return true;
      const leave = await showConfirm({
        title: 'Generation in Progress',
        message: 'Your resume is still being generated. It will continue in the background if you leave, but you won\'t see the live progress.',
        confirmLabel: 'LEAVE',
        variant: 'warning',
      });
      if (leave) {
        // Dismiss paywall if it was open (429 flow) to prevent it persisting
        if (paywallState.visible) _closePaywall(false);
      }
      return leave;
    });

    return { profileStore, selectedProfile, company, role, jobDescription, submitting, error, readyProfiles, formatDate, submit, processing, finalStatus, progress, statusLines, phaseLabel, jdLines, visibleJdLines, jdDone, retryGeneration, goToJob, reset };
  },
};

// ================================================================
// PAGES — Job Detail
// ================================================================
const JobDetailPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div class="min-w-0">
            <div class="flex items-center gap-2 text-sm font-mono min-w-0">
              <router-link to="/jobs" class="font-bold hover:text-white transition flex-shrink-0" style="color:var(--text-dim)">JOBS</router-link>
              <span style="color:var(--text-dim)" class="flex-shrink-0">/</span>
              <span class="font-bold text-white truncate">{{ breadcrumbTitle }}</span>
            </div>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Resume generation detail</p>
          </div>
        </template>
        <template #right>
          <div class="flex items-center gap-3">
            <button v-if="jobStore.current && jobStore.current.status === 'READY'" @click="doDownload" class="btn-primary text-xs" :disabled="downloading || jobStore.loading">
              <svg v-if="!downloading" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
              <span v-if="downloading" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
              <span class="hidden md:inline">{{ downloading ? 'DOWNLOADING...' : 'DOWNLOAD PDF' }}</span>
              <span class="md:hidden">{{ downloading ? '...' : 'PDF' }}</span>
            </button>
          </div>
        </template>
      </TopHeader>

      <!-- ====== READY STATE: Split layout with PDF + Chat ====== -->
      <div v-if="jobStore.current && jobStore.current.status === 'READY' && jobStore.customResume" class="split-layout">
        <div class="split-main" style="padding:0;position:relative;">
          <div v-if="pdfLoading" class="flex items-center justify-center" style="height:100%;">
            <div class="text-center">
              <div class="spinner mx-auto mb-3" style="width:28px;height:28px;border-width:3px;"></div>
              <p class="text-xs font-mono" style="color:var(--text-dim)">Loading PDF...</p>
            </div>
          </div>
          <div v-else-if="pdfError" class="flex items-center justify-center" style="height:100%;">
            <div class="text-center p-6">
              <p class="text-sm text-white mb-2">Could not load PDF preview</p>
              <p class="text-xs mb-4" style="color:var(--text-dim)">{{ pdfError }}</p>
              <button @click="loadPdf" class="btn-secondary text-xs">RETRY</button>
            </div>
          </div>
          <iframe v-else-if="pdfUrl" :src="pdfUrl" class="pdf-embed"></iframe>
          <div v-if="pdfRecompiling" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(15,27,46,0.7);border-radius:8px;z-index:5;">
            <div class="text-center">
              <div class="spinner mx-auto mb-2" style="width:24px;height:24px;border-width:2px;"></div>
              <p class="text-xs font-mono" style="color:var(--text-dim)">Recompiling PDF...</p>
            </div>
          </div>
        </div>
        <ChatPanel entity-type="job" :entity-id="jobId" :visible="chatOpen" @close="chatOpen = false" @modified="onChatModified" />
      </div>

      <!-- ====== NON-READY STATES: Standard scrollable layout ====== -->
      <div v-else class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div v-if="jobStore.loading && !jobStore.current" class="flex items-center justify-center py-16"><div class="spinner"></div></div>
        <div v-else-if="!jobStore.current" class="empty-state py-16"><p class="text-lg text-white">Job not found</p></div>
        <div v-else class="max-w-4xl mx-auto">

          <!-- ====== PROCESSING STATE ====== -->
          <div v-if="isProcessing" class="widget-card text-center py-12 mb-6 fade-slide-in">
            <div class="spinner mx-auto mb-4" style="width:32px;height:32px;border-width:3px;"></div>
            <p class="text-white font-semibold mb-1">{{ processingMessage }}</p>
            <p class="text-xs" style="color:var(--text-dim)">This page will update automatically</p>
            <div class="mx-auto mt-4 px-4 py-3 rounded-lg text-left" style="background:rgba(251,65,0,0.08); border:1px solid rgba(251,65,0,0.2); max-width:340px;">
              <div class="flex items-center gap-2">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--orange)" stroke-width="2" class="flex-shrink-0"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <p class="text-[11px]" style="color:var(--orange)">Usually 1-2 min, up to 3-4 min in rare cases. If it fails, you can retry.</p>
              </div>
            </div>
            <div class="progress-track mt-6 mx-auto" style="max-width:300px;">
              <div class="progress-fill transition-all duration-500" :style="{ width: progressWidth }"></div>
            </div>
            <div class="mt-4 space-y-1">
              <div v-for="(step, i) in progressSteps" :key="i"
                   class="boot-line visible" :class="{ ok: step.done }">
                {{ step.text }}
              </div>
            </div>
          </div>

          <!-- ====== FAILED STATE ====== -->
          <div v-if="jobStore.current.status === 'FAILED'" class="card-warning p-6 mb-6">
            <h3 class="text-white font-bold text-sm font-mono mb-2">GENERATION FAILED</h3>
            <p class="text-xs text-slate-400 mb-4">
              {{ failedInPhase2 ? 'PDF compilation failed. Your resume data is intact — you can retry the PDF build.' : 'An error occurred while generating the resume. Please try again.' }}
            </p>
            <button @click="smartRetry" class="btn-secondary text-xs" :disabled="jobStore.loading">
              {{ failedInPhase2 ? 'RETRY PDF BUILD' : 'RETRY' }}
            </button>
          </div>

          <!-- ====== PENDING STATE ====== -->
          <div v-if="jobStore.current.status === 'PENDING'" class="widget-card text-center py-12 mb-6 fade-slide-in">
            <p class="text-white font-semibold mb-4">Ready to generate a tailored resume</p>
            <p v-if="genError" class="text-xs mb-3" style="color:#ef4444;">{{ genError }}</p>
            <button @click="doGenerateResume" class="btn-primary" :disabled="generating || jobStore.loading">
              <span v-if="generating" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
              GENERATE RESUME
            </button>
          </div>

          <!-- ====== RESUME_GENERATED STATE (edge case — normally auto-chains) ====== -->
          <div v-if="jobStore.current.status === 'RESUME_GENERATED'" class="widget-card p-6 mb-6 fade-slide-in">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-white font-bold text-sm font-mono mb-1">RESUME DATA READY</h3>
                <p class="text-xs text-slate-400">Resume data generated. Build the PDF to download.</p>
              </div>
              <button @click="doGeneratePdf" class="btn-primary text-xs" :disabled="generating || jobStore.loading">
                <span v-if="generating" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
                BUILD PDF
              </button>
            </div>
          </div>

          <div class="h-8"></div>
        </div>
      </div>

      <!-- Mobile: chat backdrop + FAB -->
      <div v-if="showChatFab" class="chat-backdrop" :class="{ visible: chatOpen }" @click="chatOpen = false"></div>
      <button v-if="showChatFab && !chatOpen" class="chat-fab" @click="chatOpen = true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      </button>
    </div>
  `,
  setup() {
    const router = useRouter();
    const jobStore = useJobStore();
    const generating = ref(false);
    const downloading = ref(false);
    const chatOpen = ref(false);
    const pdfUrl = ref(null);
    const pdfLoading = ref(false);
    const pdfError = ref(null);
    const pdfRecompiling = ref(false);
    const isMobile = ref(window.innerWidth <= 768);
    const jobId = computed(() => jobStore.selectedId);

    // -- Breadcrumb --
    const breadcrumbTitle = computed(() => {
      const jd = jobStore.current?.job_description;
      if (jd && jd.role && jd.company) return `${jd.role} at ${jd.company}`;
      if (jd && jd.role) return jd.role;
      return 'Job Details';
    });

    // -- Show chat FAB only on READY --
    const showChatFab = computed(() => {
      return jobStore.current?.status === 'READY' && !!jobStore.customResume;
    });

    // -- States --
    const isProcessing = computed(() => {
      const s = jobStore.current?.status;
      return s === 'GENERATING_RESUME' || s === 'GENERATING_PDF';
    });
    const failedInPhase2 = computed(() => {
      return jobStore.current?.status === 'FAILED' && !!jobStore.current?.custom_resume_data;
    });

    // -- Progress feedback --
    const processingMessage = computed(() => {
      const s = jobStore.current?.status;
      if (s === 'GENERATING_RESUME') return 'Generating tailored resume data...';
      if (s === 'RESUME_GENERATED') return 'Resume data ready. Building PDF...';
      if (s === 'GENERATING_PDF') return 'Compiling PDF with LaTeX...';
      return 'Processing...';
    });
    const progressWidth = computed(() => {
      const s = jobStore.current?.status;
      if (s === 'GENERATING_RESUME') return '35%';
      if (s === 'RESUME_GENERATED') return '60%';
      if (s === 'GENERATING_PDF') return '80%';
      return '10%';
    });
    const progressSteps = computed(() => {
      const s = jobStore.current?.status;
      const steps = [
        { text: '[AI] Analyzing job description & profile...', done: s !== 'GENERATING_RESUME' },
        { text: '[AI] Tailoring resume content...', done: ['RESUME_GENERATED', 'GENERATING_PDF', 'READY'].includes(s) },
        { text: '[TEX] Compiling LaTeX → PDF...', done: s === 'READY' },
      ];
      return steps.filter((_, i) => {
        if (s === 'GENERATING_RESUME') return i <= 1;
        if (s === 'RESUME_GENERATED') return i <= 2;
        return true;
      });
    });

    // -- PDF blob loading --
    const loadPdf = async () => {
      if (pdfUrl.value) { URL.revokeObjectURL(pdfUrl.value); pdfUrl.value = null; }
      pdfLoading.value = true;
      pdfError.value = null;
      try {
        const token = localStorage.getItem('token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch(`/jobs/${jobId.value}/pdf?inline=true`, { headers });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        pdfUrl.value = URL.createObjectURL(blob);
      } catch (e) {
        pdfError.value = e.message || 'Failed to load PDF';
      } finally {
        pdfLoading.value = false;
      }
    };

    // -- Chat modification handler --
    // Uses local pdfRecompiling state to avoid mutating store status,
    // which would unmount the split layout and destroy the chat panel.
    const onChatModified = async (data) => {
      if (!data.custom_resume_data) return;
      if (jobStore.current) {
        jobStore.current.custom_resume_data = data.custom_resume_data;
        jobStore.customResume = data.custom_resume_data;
      }
      // Backend triggers PDF recompile — poll with a lightweight inline loop
      // that doesn't touch the store's status field.
      pdfRecompiling.value = true;
      try {
        const maxPolls = 60;
        let count = 0;
        while (count < maxPolls && !unmounted) {
          await new Promise(r => setTimeout(r, 2000));
          count++;
          try {
            const statusData = await api.get(`/jobs/${jobId.value}/status`);
            if (statusData.status === 'READY' || statusData.status === 'FAILED') {
              if (jobStore.current) jobStore.current.status = statusData.status;
              break;
            }
          } catch { break; }
        }
        if (!unmounted) await loadPdf();
      } finally {
        pdfRecompiling.value = false;
      }
    };

    // -- Load + auto-poll --
    const creditStore = useCreditStore();
    const startPolling = () => {
      jobStore.pollStatus(jobId.value).then(() => {
        if (unmounted) return;
        jobStore.fetchJob(jobId.value).then(() => {
          if (unmounted) return;
          if (jobStore.current?.status === 'READY') loadPdf();
        });
        creditStore.fetchBalance();
      });
    };

    const needsPolling = computed(() => {
      const s = jobStore.current?.status;
      return s === 'GENERATING_RESUME' || s === 'GENERATING_PDF' || s === 'RESUME_GENERATED';
    });

    const load = async () => {
      if (!jobId.value) { router.replace('/jobs'); return; }
      await jobStore.fetchJob(jobId.value);
      if (needsPolling.value) {
        startPolling();
      } else if (jobStore.current?.status === 'READY') {
        loadPdf();
      }
    };

    // -- Actions with IMMEDIATE UI feedback --
    const genError = ref(null);
    const doGenerateResume = async () => {
      generating.value = true;
      genError.value = null;
      const prevStatus = jobStore.current?.status;
      if (jobStore.current) jobStore.current.status = 'GENERATING_RESUME';
      try {
        await jobStore.generateResume(jobId.value);
        creditStore.fetchBalance();
        startPolling();
      } catch (e) {
        if (jobStore.current && prevStatus) jobStore.current.status = prevStatus;
        if (e.is429) {
          const bought = await showPaywall();
          if (bought && !unmounted) {
            try {
              if (jobStore.current) jobStore.current.status = 'GENERATING_RESUME';
              await jobStore.generateResume(jobId.value);
              creditStore.fetchBalance();
              startPolling();
            } catch (retryErr) {
              if (jobStore.current && prevStatus) jobStore.current.status = prevStatus;
              genError.value = retryErr.message || 'Generation failed';
            }
          }
        } else {
          genError.value = e.message || 'Generation failed';
        }
      }
      finally { generating.value = false; }
    };

    const doGeneratePdf = async () => {
      generating.value = true;
      const prevStatus = jobStore.current?.status;
      if (jobStore.current) jobStore.current.status = 'GENERATING_PDF';
      try {
        await jobStore.generatePdf(jobId.value);
        startPolling();
      } catch (e) {
        if (jobStore.current && prevStatus) jobStore.current.status = prevStatus;
      }
      finally { generating.value = false; }
    };

    const smartRetry = () => {
      if (failedInPhase2.value) doGeneratePdf();
      else doGenerateResume();
    };

    const doDownload = async () => {
      downloading.value = true;
      try {
        const cr = jobStore.customResume || jobStore.current?.custom_resume_data || {};
        const jd = jobStore.current?.job_description || {};
        await downloadJobPdf(jobId.value, cr.name, jd.role, jd.company);
      } catch (e) {
        alert('PDF download failed: ' + e.message);
      } finally {
        downloading.value = false;
      }
    };

    let unmounted = false;
    const onResize = () => { isMobile.value = window.innerWidth <= 768; };
    onMounted(() => { load(); window.addEventListener('resize', onResize); });

    // Cleanup blob URL + resize listener on unmount
    onUnmounted(() => {
      unmounted = true;
      window.removeEventListener('resize', onResize);
      if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value);
    });

    return { jobStore, jobId, generating, downloading, chatOpen, pdfUrl, pdfLoading, pdfError, pdfRecompiling, isMobile, genError, breadcrumbTitle, showChatFab, isProcessing, failedInPhase2, processingMessage, progressWidth, progressSteps, loadPdf, onChatModified, doGenerateResume, doGeneratePdf, smartRetry, doDownload };
  },
};

// ================================================================
// LAYOUT — App Shell (sidebar + router-view)
// ================================================================
// ================================================================
// PAGES — Admin
// ================================================================
const AdminPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <h1 class="text-sm font-bold text-white font-mono tracking-tight">ADMIN</h1>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Overview, users, catalog &amp; transactions</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">

        <!-- Tabs (horizontally scrollable on mobile) -->
        <div class="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0 mb-6">
          <div class="flex gap-2" style="min-width:max-content;">
            <button v-for="t in tabs" :key="t.key" @click="switchTab(t.key)" class="btn-ghost"
                    :style="activeTab === t.key ? 'border-color: var(--orange); color: var(--orange);' : ''">
              {{ t.label }}
            </button>
          </div>
        </div>

        <div v-if="loading" class="flex items-center justify-center py-16"><div class="spinner"></div></div>

        <!-- ====== OVERVIEW TAB ====== -->
        <div v-else-if="activeTab === 'overview'" class="fade-slide-in space-y-6">

          <!-- Section 1: KPI Grid -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div class="widget-card text-center">
              <div class="kpi-label">USERS</div>
              <div class="text-xl font-mono font-bold text-white">{{ overview.total_users }}</div>
              <div v-if="overview.new_users_today > 0" class="text-[10px] font-mono mt-1" style="color:var(--green)">+{{ overview.new_users_today }} today</div>
            </div>
            <div class="widget-card text-center" style="border-left:3px solid var(--teal);">
              <div class="kpi-label">PROFILES</div>
              <div class="text-xl font-mono font-bold" style="color:var(--teal)">{{ overview.total_profiles }}</div>
            </div>
            <div class="widget-card text-center">
              <div class="kpi-label">JOBS</div>
              <div class="text-xl font-mono font-bold text-white">{{ overview.total_jobs }}</div>
              <div class="text-[10px] font-mono mt-1" style="color:var(--green)">{{ overview.completed_jobs }} ready</div>
            </div>
            <div class="widget-card text-center" style="border-left:3px solid var(--orange);">
              <div class="kpi-label">ROASTS</div>
              <div class="text-xl font-mono font-bold" style="color:var(--orange)">{{ overview.total_roasts }}</div>
            </div>
            <div class="widget-card text-center" style="border-left:3px solid var(--orange);">
              <div class="kpi-label">PURCHASES</div>
              <div class="text-xl font-mono font-bold" style="color:var(--orange)">{{ overview.total_purchase_txns }}</div>
            </div>
            <div class="widget-card text-center">
              <div class="kpi-label">USED TODAY</div>
              <div class="text-xl font-mono font-bold text-white">{{ overview.consumed_today }}</div>
            </div>
            <div class="widget-card text-center" style="border-left:3px solid var(--teal);">
              <div class="kpi-label">ACTIVE PASSES</div>
              <div class="text-xl font-mono font-bold" style="color:var(--teal)">{{ overview.active_time_passes }}</div>
            </div>
            <div class="widget-card text-center" style="border-left:3px solid var(--green);">
              <div class="kpi-label">NEW (7D)</div>
              <div class="text-xl font-mono font-bold" style="color:var(--green)">{{ overview.new_users_7d }}</div>
            </div>
          </div>

          <!-- Section 2: 7-Day Trends -->
          <div>
            <div class="section-label mb-3">7-Day Trends</div>
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div v-for="(series, idx) in trendSeries" :key="idx" class="widget-card">
                <div class="flex items-center justify-between mb-2">
                  <span class="kpi-label">{{ series.title }}</span>
                  <span class="font-mono text-xs font-bold" :style="{ color: series.color }">{{ series.total }}</span>
                </div>
                <div class="flex items-end gap-1" style="height:80px;">
                  <div v-for="(val, i) in series.data" :key="i" class="flex-1 flex flex-col items-center justify-end" style="height:100%;">
                    <span v-if="val > 0" class="text-[8px] font-mono font-bold leading-none mb-0.5" :style="{ color: series.color }">{{ val }}</span>
                    <div :style="{ height: barHeight(val, series.data) + 'px', background: series.color, borderRadius: '2px 2px 0 0', width: '100%', minWidth: '4px', opacity: val > 0 ? 1 : 0.3 }"></div>
                    <span class="text-[8px] font-mono leading-none mt-1" style="color:var(--text-dim)">{{ trendDayLabels[i] }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Section 3: Conversion Funnel -->
          <div>
            <div class="section-label mb-3">Conversion Funnel</div>
            <div class="widget-card space-y-3">
              <div v-for="(step, idx) in funnelSteps" :key="idx">
                <div class="flex items-center justify-between mb-1">
                  <span class="text-xs font-mono font-medium" style="color:var(--text-secondary)">{{ step.label }}</span>
                  <div class="flex items-center gap-2">
                    <span class="font-mono text-sm font-bold text-white">{{ step.count }}</span>
                    <span v-if="idx > 0 && funnelSteps[idx-1].count > 0" class="text-[10px] font-mono" style="color:var(--text-dim)">
                      ({{ Math.round((step.count / funnelSteps[idx-1].count) * 100) }}%)
                    </span>
                    <span v-else-if="idx === 0" class="text-[10px] font-mono" style="color:var(--text-dim)">(100%)</span>
                  </div>
                </div>
                <div class="progress-track">
                  <div class="progress-fill" :style="{ width: (overview.funnel.users > 0 ? (step.count / overview.funnel.users) * 100 : 0) + '%' }"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Section 4: Status Breakdowns -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="widget-card">
              <div class="kpi-label mb-3">Job Status</div>
              <div v-if="Object.keys(overview.job_status_breakdown).length === 0" class="text-xs font-mono" style="color:var(--text-dim)">No jobs yet</div>
              <template v-else>
                <div class="flex rounded overflow-hidden mb-3" style="height:12px;">
                  <div v-for="(count, status) in overview.job_status_breakdown" :key="status"
                       :style="{ width: (overview.total_jobs > 0 ? (count / overview.total_jobs) * 100 : 0) + '%', background: statusColor(status) }"
                       :title="status + ': ' + count"></div>
                </div>
                <div class="flex flex-wrap gap-2">
                  <span v-for="(count, status) in overview.job_status_breakdown" :key="status"
                        class="inline-flex items-center gap-1.5 text-[10px] font-mono" style="color:var(--text-secondary)">
                    <span class="w-2 h-2 rounded-sm" :style="{ background: statusColor(status) }"></span>
                    {{ status }} <span class="font-bold">{{ count }}</span>
                  </span>
                </div>
              </template>
            </div>
            <div class="widget-card">
              <div class="kpi-label mb-3">Roast Status</div>
              <div v-if="Object.keys(overview.roast_status_breakdown).length === 0" class="text-xs font-mono" style="color:var(--text-dim)">No roasts yet</div>
              <template v-else>
                <div class="flex rounded overflow-hidden mb-3" style="height:12px;">
                  <div v-for="(count, status) in overview.roast_status_breakdown" :key="status"
                       :style="{ width: (overview.total_roasts > 0 ? (count / overview.total_roasts) * 100 : 0) + '%', background: statusColor(status) }"
                       :title="status + ': ' + count"></div>
                </div>
                <div class="flex flex-wrap gap-2">
                  <span v-for="(count, status) in overview.roast_status_breakdown" :key="status"
                        class="inline-flex items-center gap-1.5 text-[10px] font-mono" style="color:var(--text-secondary)">
                    <span class="w-2 h-2 rounded-sm" :style="{ background: statusColor(status) }"></span>
                    {{ status }} <span class="font-bold">{{ count }}</span>
                  </span>
                </div>
              </template>
            </div>
          </div>

          <!-- Section 5: LLM Usage -->
          <div>
            <div class="section-label mb-3">LLM Usage</div>
            <div v-if="overview.llm_summary.total_requests === 0" class="widget-card">
              <p class="text-xs font-mono" style="color:var(--text-dim)">No LLM requests yet</p>
            </div>
            <template v-else>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
                <div class="widget-card text-center">
                  <div class="kpi-label">REQUESTS</div>
                  <div class="text-lg font-mono font-bold text-white">{{ overview.llm_summary.total_requests }}</div>
                </div>
                <div class="widget-card text-center">
                  <div class="kpi-label">TOKENS (IN/OUT)</div>
                  <div class="text-lg font-mono font-bold text-white">{{ formatTokens(overview.llm_summary.total_input_tokens) }} / {{ formatTokens(overview.llm_summary.total_output_tokens) }}</div>
                </div>
                <div class="widget-card text-center">
                  <div class="kpi-label">AVG RESPONSE</div>
                  <div class="text-lg font-mono font-bold text-white">{{ (overview.llm_summary.avg_response_time_ms / 1000).toFixed(1) }}s</div>
                </div>
                <div class="widget-card text-center">
                  <div class="kpi-label">SUCCESS RATE</div>
                  <div class="text-lg font-mono font-bold" :style="{ color: overview.llm_summary.success_rate_pct >= 95 ? 'var(--green)' : 'var(--amber)' }">{{ overview.llm_summary.success_rate_pct }}%</div>
                </div>
              </div>
              <div class="widget-card">
                <div class="overflow-x-auto scroll-hint">
                  <table class="data-table admin-table">
                    <thead><tr>
                      <th>Model</th><th>Requests</th><th>Input</th><th>Output</th>
                      <th class="hidden md:table-cell">Cached</th>
                      <th class="hidden md:table-cell">Avg ms</th>
                      <th>Est. Cost</th>
                    </tr></thead>
                    <tbody>
                      <tr v-for="m in overview.llm_summary.by_model" :key="m.model_name">
                        <td class="font-mono text-white text-[11px]">{{ m.model_name }}</td>
                        <td class="font-mono">{{ m.request_count }}</td>
                        <td class="font-mono">{{ formatTokens(m.input_tokens) }}</td>
                        <td class="font-mono">{{ formatTokens(m.output_tokens) }}</td>
                        <td class="font-mono hidden md:table-cell">{{ formatTokens(m.cached_tokens) }}</td>
                        <td class="font-mono hidden md:table-cell">{{ m.avg_response_time_ms }}</td>
                        <td class="font-mono font-bold" style="color:var(--orange)">\${{ m.estimated_cost_usd.toFixed(3) }}</td>
                      </tr>
                    </tbody>
                    <tfoot>
                      <tr><td colspan="6" class="text-right font-mono text-[10px]" style="color:var(--text-dim)">TOTAL EST. COST</td>
                        <td class="font-mono font-bold" style="color:var(--orange)">\${{ overview.llm_summary.total_estimated_cost_usd.toFixed(3) }}</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </div>
            </template>
          </div>

          <!-- Section 6: Recent Activity -->
          <div>
            <div class="section-label mb-3">Recent Activity</div>
            <div class="widget-card">
              <div v-if="!overview.recent_activity || overview.recent_activity.length === 0" class="empty-state"><p class="text-sm">No recent activity</p></div>
              <div v-else class="space-y-2">
                <div v-for="a in overview.recent_activity" :key="a.id" class="flex items-center justify-between py-2 px-3 rounded-lg" style="background:rgba(0,0,0,0.15);">
                  <div class="flex items-center gap-3">
                    <span :style="{ color: a.amount >= 0 ? 'var(--green)' : 'var(--red)' }" class="font-mono font-bold text-sm w-10 text-right">{{ a.amount >= 0 ? '+' : '' }}{{ a.amount }}</span>
                    <div>
                      <span class="text-white text-xs">{{ a.user_email }}</span>
                      <span class="text-[10px] ml-2 font-mono" style="color:var(--text-dim)">{{ a.type }}</span>
                    </div>
                  </div>
                  <span class="text-[10px] font-mono" style="color:var(--text-dim)">{{ formatDate(a.created_at, { includeTime: true }) }}</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        <!-- ====== USERS TAB ====== -->
        <div v-else-if="activeTab === 'users'" class="fade-slide-in">
          <div class="flex items-center gap-3 mb-4">
            <input v-model="userSearch" class="input-field input-mono flex-1" placeholder="Search by name or email..." @keyup.enter="loadUsers(1)" />
            <button @click="loadUsers(1)" class="btn-ghost">SEARCH</button>
          </div>
          <p v-if="userError" class="text-red-400 text-xs mb-4 font-mono">{{ userError }}</p>
          <div class="widget-card">
            <div v-if="users.length === 0" class="empty-state"><p class="text-sm">No users</p></div>
            <div class="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0" v-else>
            <table class="data-table admin-table" style="min-width:780px;">
              <thead><tr><th>Name</th><th>Email</th><th>Credits</th><th>Free</th><th>Pass</th><th>Jobs</th><th class="hidden md:table-cell">Tenant</th><th class="hidden md:table-cell">Role</th><th></th></tr></thead>
              <tbody>
                <template v-for="u in users" :key="u.id">
                  <tr @click="toggleUserDetail(u.id)" style="cursor:pointer;" :style="expandedUser === u.id ? 'background:rgba(251,65,0,0.04);' : ''">
                    <td class="text-white font-medium">{{ u.name }}</td>
                    <td class="font-mono text-[11px]" style="color:var(--text-dim)">{{ u.email }}</td>
                    <td class="font-mono font-bold" style="color:var(--orange)">{{ u.balance }}</td>
                    <td class="font-mono text-xs">{{ u.daily_free_remaining }}</td>
                    <td>
                      <span v-if="u.active_time_pass" class="badge badge-ready">{{ u.active_time_pass.tier_name }}</span>
                      <span v-else class="text-[10px] font-mono" style="color:var(--text-dim)">-</span>
                    </td>
                    <td class="font-mono text-xs">{{ u.job_count }}</td>
                    <td class="hidden md:table-cell">
                      <select v-model="u.tenant_id" @change="assignTenant(u.id, u.tenant_id)" @click.stop
                              class="input-field" style="padding:4px 24px 4px 8px; width:auto; font-size:11px;" :disabled="saving">
                        <option :value="null">--</option>
                        <option v-for="t in tenants" :key="t.id" :value="t.id">{{ t.name }}</option>
                      </select>
                    </td>
                    <td class="hidden md:table-cell">
                      <span v-if="u.is_super_admin" class="badge" style="background:rgba(251,65,0,0.15); color:var(--orange); font-size:9px;">ADMIN</span>
                      <span v-else class="badge" style="background:rgba(148,163,184,0.1); color:var(--text-dim); font-size:9px;">USER</span>
                    </td>
                    <td class="text-right">
                      <svg :style="{ transform: expandedUser === u.id ? 'rotate(180deg)' : '' }" style="transition:transform 0.2s;" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
                    </td>
                  </tr>
                  <!-- Expanded user detail row -->
                  <tr v-if="expandedUser === u.id">
                    <td colspan="9" style="padding:0; border:none;">
                      <div class="p-4" style="background:rgba(0,0,0,0.15); border-top:1px solid rgba(255,255,255,0.04);">
                        <div v-if="userDetailLoading" class="flex justify-center py-4"><div class="loading-spinner"></div></div>
                        <template v-else-if="userDetail">
                          <!-- Summary cards -->
                          <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                            <div class="p-3 rounded-lg" style="background:rgba(0,0,0,0.2);">
                              <div class="kpi-label">BALANCE</div>
                              <div class="text-lg font-mono font-bold" style="color:var(--orange)">{{ userDetail.balance.balance }}</div>
                            </div>
                            <div class="p-3 rounded-lg" style="background:rgba(0,0,0,0.2);">
                              <div class="kpi-label">DAILY FREE</div>
                              <div class="text-lg font-mono font-bold text-white">{{ userDetail.balance.daily_free_remaining }} / {{ userDetail.balance.daily_free_total }}</div>
                            </div>
                            <div class="p-3 rounded-lg" style="background:rgba(0,0,0,0.2);">
                              <div class="kpi-label">TIME PASS</div>
                              <div v-if="userDetail.balance.active_time_pass" class="text-sm font-mono" style="color:var(--teal)">
                                {{ userDetail.balance.active_time_pass.tier_name }}
                                <div class="text-[10px]" style="color:var(--text-dim)">expires {{ formatDate(userDetail.balance.active_time_pass.expires_at) }}</div>
                              </div>
                              <div v-else class="text-sm font-mono" style="color:var(--text-dim)">None</div>
                            </div>
                            <div class="p-3 rounded-lg" style="background:rgba(0,0,0,0.2);">
                              <div class="kpi-label">GRANT CREDITS</div>
                              <div class="flex items-center gap-2 mt-1">
                                <input v-model.number="inlineGrantAmount" type="number" class="input-field" style="width:70px; padding:4px 8px; font-size:12px;" placeholder="qty" />
                                <button @click="inlineGrant(u.id)" class="btn-ghost text-xs" :disabled="!inlineGrantAmount || saving" style="padding:4px 10px;">GRANT</button>
                              </div>
                              <p v-if="inlineGrantMsg" class="text-[10px] mt-1 font-mono" :style="{ color: inlineGrantError ? 'var(--red)' : 'var(--green)' }">{{ inlineGrantMsg }}</p>
                            </div>
                          </div>
                          <!-- Transaction history -->
                          <div class="section-label mb-2">Transaction History</div>
                          <div v-if="userDetail.transactions.items.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No transactions</div>
                          <div v-else class="overflow-x-auto">
                            <table class="data-table" style="min-width:500px;">
                              <thead><tr><th>Date</th><th>Type</th><th>Amount</th><th>Description</th></tr></thead>
                              <tbody>
                                <tr v-for="t in userDetail.transactions.items" :key="t.id">
                                  <td class="font-mono text-[10px]" style="color:var(--text-dim)">{{ formatDate(t.created_at, { includeTime: true }) }}</td>
                                  <td><span class="badge badge-processing" style="font-size:9px;">{{ t.type }}</span></td>
                                  <td :style="{ color: t.amount >= 0 ? 'var(--green)' : 'var(--red)' }" class="font-mono font-bold text-xs">{{ t.amount >= 0 ? '+' : '' }}{{ t.amount }}</td>
                                  <td class="text-[10px]" style="color:var(--text-dim)">{{ t.description || '-' }}</td>
                                </tr>
                              </tbody>
                            </table>
                            <div v-if="userDetail.transactions.pages > 1" class="flex items-center justify-center gap-3 mt-3">
                              <button class="btn-ghost text-xs" :disabled="userDetailPage <= 1" @click="loadUserDetail(u.id, userDetailPage - 1)">PREV</button>
                              <span class="font-mono text-[10px] text-slate-400">{{ userDetailPage }} / {{ userDetail.transactions.pages }}</span>
                              <button class="btn-ghost text-xs" :disabled="userDetailPage >= userDetail.transactions.pages" @click="loadUserDetail(u.id, userDetailPage + 1)">NEXT</button>
                            </div>
                          </div>
                        </template>
                      </div>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
            </div>
            <div v-if="userPages > 1" class="flex items-center justify-center gap-4 mt-4">
              <button class="btn-ghost text-xs" :disabled="userPage <= 1" @click="loadUsers(userPage - 1)">PREV</button>
              <span class="font-mono text-xs text-slate-400">{{ userPage }} / {{ userPages }}</span>
              <button class="btn-ghost text-xs" :disabled="userPage >= userPages" @click="loadUsers(userPage + 1)">NEXT</button>
            </div>
          </div>
        </div>

        <!-- ====== CATALOG TAB ====== -->
        <div v-else-if="activeTab === 'catalog'" class="fade-slide-in">
          <!-- Credit Packs section -->
          <div class="section-label mb-4">Credit Packs</div>
          <div class="widget-card mb-6">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end mb-4">
              <div><label class="kpi-label block mb-1">Name</label><input v-model="packForm.name" class="input-field" placeholder="e.g. Starter Pack" :disabled="saving" /></div>
              <div><label class="kpi-label block mb-1">Credits</label><input v-model.number="packForm.credits" type="number" class="input-field" :disabled="saving" /></div>
              <div><label class="kpi-label block mb-1">Price (paise)</label><input v-model.number="packForm.price_paise" type="number" class="input-field" :disabled="saving" /></div>
              <button @click="createPack" class="btn-ghost" :disabled="!packForm.name || !packForm.credits || !packForm.price_paise || saving">+ CREATE</button>
            </div>
            <p v-if="packError" class="text-red-400 text-xs mb-3 font-mono">{{ packError }}</p>
            <div v-if="creditPacks.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No credit packs yet</div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table">
                <thead><tr><th>Name</th><th>Credits</th><th>Price</th><th>Active</th><th>Order</th><th class="text-right">Actions</th></tr></thead>
                <tbody>
                  <tr v-for="p in creditPacks" :key="p.id">
                    <td class="text-white font-medium">{{ p.name }}</td>
                    <td class="font-mono">{{ p.credits }}</td>
                    <td class="font-mono">\u20B9{{ (p.price_paise / 100).toFixed(0) }}</td>
                    <td><span :class="p.is_active ? 'badge-ready' : 'badge-failed'" class="badge">{{ p.is_active ? 'YES' : 'NO' }}</span></td>
                    <td class="font-mono text-[11px]">{{ p.sort_order }}</td>
                    <td class="text-right">
                      <button @click="togglePack(p)" class="btn-ghost text-xs mr-1" :disabled="saving">{{ p.is_active ? 'DEACTIVATE' : 'ACTIVATE' }}</button>
                      <button @click="deletePack(p.id)" class="btn-ghost danger text-xs" :disabled="saving">DELETE</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Time Passes section -->
          <div class="section-label mb-4">Time Passes <span class="text-[9px] font-mono ml-1 px-1.5 py-0.5 rounded" style="background:rgba(1,169,219,0.15);color:var(--teal);">COMING SOON</span></div>
          <div class="widget-card mb-6">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end mb-4">
              <div><label class="kpi-label block mb-1">Name</label><input v-model="tierForm.name" class="input-field" placeholder="e.g. 7-Day Pass" :disabled="saving" /></div>
              <div><label class="kpi-label block mb-1">Days</label><input v-model.number="tierForm.duration_days" type="number" class="input-field" :disabled="saving" /></div>
              <div><label class="kpi-label block mb-1">Price (paise)</label><input v-model.number="tierForm.price_paise" type="number" class="input-field" :disabled="saving" /></div>
              <button @click="createTier" class="btn-ghost" :disabled="!tierForm.name || !tierForm.duration_days || !tierForm.price_paise || saving">+ CREATE</button>
            </div>
            <p v-if="tierError" class="text-red-400 text-xs mb-3 font-mono">{{ tierError }}</p>
            <div v-if="timePassTiers.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No time pass tiers yet</div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table">
                <thead><tr><th>Name</th><th>Days</th><th>Price</th><th>Active</th><th class="text-right">Actions</th></tr></thead>
                <tbody>
                  <tr v-for="t in timePassTiers" :key="t.id">
                    <td class="text-white font-medium">{{ t.name }}</td>
                    <td class="font-mono">{{ t.duration_days }}</td>
                    <td class="font-mono">\u20B9{{ (t.price_paise / 100).toFixed(0) }}</td>
                    <td><span :class="t.is_active ? 'badge-ready' : 'badge-failed'" class="badge">{{ t.is_active ? 'YES' : 'NO' }}</span></td>
                    <td class="text-right">
                      <button @click="toggleTier(t)" class="btn-ghost text-xs mr-1" :disabled="saving">{{ t.is_active ? 'DEACTIVATE' : 'ACTIVATE' }}</button>
                      <button @click="deleteTier(t.id)" class="btn-ghost danger text-xs" :disabled="saving">DELETE</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Promo Codes section -->
          <div class="section-label mb-4">Promo Codes</div>
          <div class="widget-card">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
              <div><label class="kpi-label block mb-1">Code</label><input v-model="promoForm.code" class="input-field input-mono" placeholder="e.g. LAUNCH50" :disabled="saving" /></div>
              <div><label class="kpi-label block mb-1">Type</label>
                <select v-model="promoForm.type" class="input-field" :disabled="saving">
                  <option value="CREDITS">Credits</option><option value="TIME_PASS">Time Pass</option>
                </select>
              </div>
              <div><label class="kpi-label block mb-1">Value</label><input v-model.number="promoForm.value" type="number" class="input-field" :disabled="saving" placeholder="credits or tier_id" /></div>
              <button @click="createPromo" class="btn-ghost" :disabled="!promoForm.code || !promoForm.value || saving">+ CREATE</button>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3 mb-4">
              <div><label class="kpi-label block mb-1">Max Redemptions (0=unlimited)</label><input v-model.number="promoForm.max_redemptions" type="number" class="input-field" :disabled="saving" /></div>
            </div>
            <p v-if="promoError" class="text-red-400 text-xs mb-3 font-mono">{{ promoError }}</p>
            <div v-if="promoCodes.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No promo codes yet</div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table">
                <thead><tr><th>Code</th><th>Type</th><th>Value</th><th>Used</th><th>Max</th><th>Active</th><th class="text-right">Actions</th></tr></thead>
                <tbody>
                  <tr v-for="p in promoCodes" :key="p.id">
                    <td class="font-mono text-white">{{ p.code }}</td>
                    <td><span class="badge badge-processing">{{ p.type }}</span></td>
                    <td class="font-mono">{{ p.value }}</td>
                    <td class="font-mono">{{ p.current_redemptions }}</td>
                    <td class="font-mono">{{ p.max_redemptions || '\u221E' }}</td>
                    <td><span :class="p.is_active ? 'badge-ready' : 'badge-failed'" class="badge">{{ p.is_active ? 'YES' : 'NO' }}</span></td>
                    <td class="text-right">
                      <button @click="togglePromo(p)" class="btn-ghost text-xs mr-1" :disabled="saving">{{ p.is_active ? 'DEACTIVATE' : 'ACTIVATE' }}</button>
                      <button @click="deletePromo(p.id)" class="btn-ghost danger text-xs" :disabled="saving">DELETE</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- ====== TRANSACTIONS TAB ====== -->
        <div v-else-if="activeTab === 'transactions'" class="fade-slide-in">
          <div class="widget-card">
            <div class="flex items-center gap-3 mb-4">
              <input v-model="txnSearch" class="input-field input-mono flex-1" placeholder="Search by description, email, razorpay ID..." @keyup.enter="loadTransactions(1)" />
              <button @click="loadTransactions(1)" class="btn-ghost">SEARCH</button>
            </div>
            <div v-if="adminTxns.length === 0" class="empty-state"><p class="text-sm">No transactions</p></div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table" style="min-width:700px;">
                <thead><tr><th>Date</th><th>User</th><th>Type</th><th>Amount</th><th>Description</th><th class="hidden md:table-cell">Razorpay</th></tr></thead>
                <tbody>
                  <tr v-for="t in adminTxns" :key="t.id">
                    <td class="font-mono text-[11px]" style="color:var(--text-dim)">{{ formatDate(t.created_at, { includeTime: true }) }}</td>
                    <td class="text-[11px]" style="cursor:pointer; color:var(--teal);" @click="jumpToUser(t.user_email)">{{ t.user_email }}</td>
                    <td><span class="badge badge-processing">{{ t.type }}</span></td>
                    <td :style="{ color: t.amount >= 0 ? 'var(--green)' : 'var(--red)' }" class="font-mono font-bold">{{ t.amount >= 0 ? '+' : '' }}{{ t.amount }}</td>
                    <td class="text-[11px]" style="color:var(--text-dim)">{{ t.description || '-' }}</td>
                    <td class="font-mono text-[10px] hidden md:table-cell" style="color:var(--text-dim)">{{ t.razorpay_order_id || '-' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-if="adminTxnPages > 1" class="flex items-center justify-center gap-4 mt-4">
              <button class="btn-ghost text-xs" :disabled="adminTxnPage <= 1" @click="loadTransactions(adminTxnPage - 1)">PREV</button>
              <span class="font-mono text-xs text-slate-400">{{ adminTxnPage }} / {{ adminTxnPages }}</span>
              <button class="btn-ghost text-xs" :disabled="adminTxnPage >= adminTxnPages" @click="loadTransactions(adminTxnPage + 1)">NEXT</button>
            </div>
          </div>
        </div>

        <!-- ====== SETTINGS TAB ====== -->
        <div v-else-if="activeTab === 'settings'" class="fade-slide-in">
          <!-- Tenants -->
          <div class="section-label mb-4">Tenants</div>
          <div class="widget-card mb-6">
            <div class="flex items-center gap-3 mb-4">
              <input v-model="newTenantName" placeholder="Tenant name" class="input-field flex-1"
                     @keyup.enter="createTenant" :disabled="saving" />
              <button @click="createTenant" class="btn-ghost" style="white-space:nowrap;"
                      :disabled="!newTenantName.trim() || saving">+ CREATE</button>
            </div>
            <p v-if="tenantError" class="text-red-400 text-xs mb-3 font-mono">{{ tenantError }}</p>
            <div v-if="tenants.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No tenants yet</div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table" style="min-width:480px;">
                <thead><tr><th>Name</th><th>Users</th><th>Created</th><th class="text-right">Actions</th></tr></thead>
                <tbody>
                  <tr v-for="t in tenants" :key="t.id">
                    <td>
                      <div v-if="editing === t.id" class="flex items-center gap-2">
                        <input v-model="editName" class="input-field" @keyup.enter="renameTenant(t.id)" @keyup.esc="editing = null" :disabled="saving" />
                      </div>
                      <span v-else class="text-white font-medium">{{ t.name }}</span>
                    </td>
                    <td><span class="badge badge-processing">{{ t.user_count }}</span></td>
                    <td class="font-mono text-[11px]" style="color:var(--text-dim)">{{ formatDate(t.created_at) }}</td>
                    <td class="text-right">
                      <template v-if="editing === t.id">
                        <button @click="renameTenant(t.id)" class="btn-ghost text-xs" :disabled="saving">SAVE</button>
                        <button @click="editing = null; renameError = null" class="btn-ghost text-xs ml-1">CANCEL</button>
                      </template>
                      <template v-else>
                        <button @click="editing = t.id; editName = t.name; renameError = null" class="btn-ghost text-xs">RENAME</button>
                        <button @click="deleteTenant(t.id, t.name)" class="btn-ghost danger text-xs ml-1" :disabled="saving">DELETE</button>
                      </template>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p v-if="renameError" class="text-red-400 text-xs mt-2 font-mono">{{ renameError }}</p>
          </div>

          <!-- Domain Rules -->
          <div class="section-label mb-4">Domain Rules</div>
          <div class="widget-card">
            <div v-if="!tenants.length" class="text-sm py-2 font-mono" style="color:var(--text-dim)">Create a tenant first.</div>
            <template v-else>
              <div class="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] gap-3 items-end mb-4">
                <div><label class="kpi-label block mb-1">Domain</label><input v-model="newRuleDomain" placeholder="e.g. google.com" class="input-field input-mono" @keyup.enter="createRule" :disabled="saving" /></div>
                <div><label class="kpi-label block mb-1">Tenant</label>
                  <select v-model="newRuleTenantId" class="input-field" style="padding:10px 28px 10px 14px;" :disabled="saving">
                    <option :value="null" disabled>Select tenant</option>
                    <option v-for="t in tenants" :key="t.id" :value="t.id">{{ t.name }}</option>
                  </select>
                </div>
                <button @click="createRule" class="btn-ghost" :disabled="!newRuleDomain.trim() || !newRuleTenantId || saving">+ CREATE</button>
              </div>
            </template>
            <p v-if="ruleError" class="text-red-400 text-xs mb-3 font-mono">{{ ruleError }}</p>
            <div v-if="rules.length === 0" class="text-xs font-mono py-2" style="color:var(--text-dim)">No domain rules</div>
            <div class="overflow-x-auto" v-else>
              <table class="data-table admin-table" style="min-width:480px;">
                <thead><tr><th>Domain</th><th>Tenant</th><th>Created</th><th class="text-right">Actions</th></tr></thead>
                <tbody>
                  <tr v-for="r in rules" :key="r.id">
                    <td class="font-mono text-sm text-white">{{ r.domain }}</td>
                    <td>{{ r.tenant_name }}</td>
                    <td class="font-mono text-[11px]" style="color:var(--text-dim)">{{ formatDate(r.created_at) }}</td>
                    <td class="text-right">
                      <button @click="deleteRule(r.id, r.domain)" class="btn-ghost danger text-xs" :disabled="saving">DELETE</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  `,
  setup() {
    const tabs = [
      { key: 'overview', label: 'OVERVIEW' },
      { key: 'users', label: 'USERS' },
      { key: 'catalog', label: 'CATALOG' },
      { key: 'transactions', label: 'TRANSACTIONS' },
      { key: 'settings', label: 'SETTINGS' },
    ];
    const activeTab = ref('overview');
    const loading = ref(true);
    const saving = ref(false);

    // ── Overview ──
    const overview = ref({
      total_users: 0, total_jobs: 0, completed_jobs: 0, total_purchase_txns: 0,
      active_time_passes: 0, consumed_today: 0, total_profiles: 0, total_roasts: 0,
      new_users_today: 0, new_users_7d: 0,
      job_status_breakdown: {}, roast_status_breakdown: {},
      funnel: { users: 0, profiles: 0, jobs: 0, ready_jobs: 0 },
      llm_summary: { total_requests: 0, total_input_tokens: 0, total_output_tokens: 0, total_cached_tokens: 0, avg_response_time_ms: 0, success_rate_pct: 100, total_estimated_cost_usd: 0, by_model: [] },
      trends: { labels: [], users: [], profiles: [], jobs: [], roasts: [] },
      recent_activity: [],
    });

    async function loadOverview() {
      try { overview.value = await api.get('/admin/overview'); } catch(e) { userError.value = 'Failed to load overview: ' + e.message; }
    }

    // ── Overview helpers ──
    function formatTokens(n) {
      if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
      if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
      return String(n);
    }

    function barHeight(val, series) {
      const max = Math.max(...series, 1);
      return Math.max(Math.round((Math.sqrt(val) / Math.sqrt(max)) * 48), 2);
    }

    const statusColorMap = { READY: 'var(--green)', FAILED: 'var(--red)', PENDING: 'var(--amber)', PROCESSING: 'var(--teal)', GENERATING_RESUME: 'var(--teal)', RESUME_GENERATED: 'var(--teal)', GENERATING_PDF: 'var(--teal)' };
    function statusColor(status) { return statusColorMap[status] || 'var(--text-dim)'; }

    const funnelSteps = computed(() => {
      const f = overview.value.funnel;
      return [
        { label: 'SIGNED UP', count: f.users },
        { label: 'UPLOADED PROFILE', count: f.profiles },
        { label: 'CREATED JOB', count: f.jobs },
        { label: 'GOT PDF', count: f.ready_jobs },
      ];
    });

    const trendSeries = computed(() => {
      const t = overview.value.trends;
      return [
        { title: 'SIGNUPS', data: t.users, color: 'var(--text-primary)', total: t.users.reduce((a, b) => a + b, 0) },
        { title: 'PROFILES', data: t.profiles, color: 'var(--teal)', total: t.profiles.reduce((a, b) => a + b, 0) },
        { title: 'JOBS', data: t.jobs, color: 'var(--text-primary)', total: t.jobs.reduce((a, b) => a + b, 0) },
        { title: 'ROASTS', data: t.roasts, color: 'var(--orange)', total: t.roasts.reduce((a, b) => a + b, 0) },
      ];
    });

    const trendDayLabels = computed(() => {
      return overview.value.trends.labels.map(l => l.split(' ')[1] || l);
    });

    // ── Users ──
    const users = ref([]);
    const userSearch = ref('');
    const userPage = ref(1);
    const userPages = ref(1);
    const userError = ref(null);
    const expandedUser = ref(null);
    const userDetail = ref(null);
    const userDetailLoading = ref(false);
    const userDetailPage = ref(1);
    const inlineGrantAmount = ref(null);
    const inlineGrantMsg = ref(null);
    const inlineGrantError = ref(false);

    async function loadUsers(page = 1) {
      try {
        const d = await api.get(`/admin/users?page=${page}&size=50&search=${encodeURIComponent(userSearch.value)}`);
        users.value = d.items; userPage.value = d.page; userPages.value = d.pages;
      } catch(e) { userError.value = 'Failed to load users: ' + e.message; }
    }

    async function toggleUserDetail(userId) {
      if (expandedUser.value === userId) { expandedUser.value = null; userDetail.value = null; return; }
      expandedUser.value = userId;
      userDetailPage.value = 1;
      inlineGrantAmount.value = null;
      inlineGrantMsg.value = null;
      inlineGrantError.value = false;
      await loadUserDetail(userId, 1);
    }

    async function loadUserDetail(userId, page = 1) {
      userDetailLoading.value = true;
      try {
        userDetail.value = await api.get(`/admin/credits/user/${userId}?page=${page}&size=10`);
        userDetailPage.value = page;
      } catch(e) { console.error(e); }
      userDetailLoading.value = false;
    }

    async function inlineGrant(userId) {
      if (!inlineGrantAmount.value || saving.value) return;
      saving.value = true;
      inlineGrantMsg.value = null; inlineGrantError.value = false;
      try {
        const d = await api.post('/admin/credits/grant', { user_id: userId, amount: inlineGrantAmount.value, description: 'Admin grant' });
        inlineGrantMsg.value = `Granted! New balance: ${d.new_balance}`;
        inlineGrantAmount.value = null;
        await loadUserDetail(userId, userDetailPage.value);
        await loadUsers(userPage.value);
      } catch(e) { inlineGrantMsg.value = e.message; inlineGrantError.value = true; }
      saving.value = false;
    }

    function jumpToUser(email) {
      activeTab.value = 'users';
      userSearch.value = email;
      loadUsers(1);
    }

    // ── Tenants ──
    const tenants = ref([]);
    const newTenantName = ref('');
    const tenantError = ref(null);
    const editing = ref(null);
    const editName = ref('');
    const renameError = ref(null);

    async function loadTenants() {
      try { const d = await api.get('/admin/tenants'); tenants.value = d.items || d; } catch(e) { console.error(e); }
    }

    async function createTenant() {
      tenantError.value = null;
      if (!newTenantName.value.trim() || saving.value) return;
      saving.value = true;
      try { await api.post('/admin/tenants', { name: newTenantName.value.trim() }); newTenantName.value = ''; await loadTenants(); } catch(e) { tenantError.value = e.message; }
      saving.value = false;
    }

    async function renameTenant(id) {
      renameError.value = null;
      if (!editName.value.trim() || saving.value) return;
      saving.value = true;
      try { await api.put('/admin/tenants/' + id, { name: editName.value.trim() }); editing.value = null; await loadTenants(); } catch(e) { renameError.value = e.message; }
      saving.value = false;
    }

    async function deleteTenant(id, name) {
      const ok = await showConfirm({ title: 'Delete Tenant', message: 'Delete "' + name + '"? Users will be unassigned.' });
      if (!ok) return; saving.value = true; tenantError.value = null;
      try { await api.del('/admin/tenants/' + id); await loadTenants(); } catch(e) { tenantError.value = e.message; }
      saving.value = false;
    }

    async function assignTenant(userId, tenantId) {
      if (saving.value) return; saving.value = true; userError.value = null;
      try { await api.put('/admin/users/' + userId + '/tenant', { tenant_id: tenantId }); } catch(e) { userError.value = e.message; await loadUsers(userPage.value); }
      saving.value = false;
    }

    // ── Domain Rules ──
    const rules = ref([]);
    const newRuleDomain = ref('');
    const newRuleTenantId = ref(null);
    const ruleError = ref(null);

    async function loadRules() {
      try { const d = await api.get('/admin/domain-rules'); rules.value = d.items || d; } catch(e) { console.error(e); }
    }

    async function createRule() {
      ruleError.value = null;
      if (!newRuleDomain.value.trim() || !newRuleTenantId.value || saving.value) return;
      saving.value = true;
      try { await api.post('/admin/domain-rules', { domain: newRuleDomain.value.trim(), tenant_id: newRuleTenantId.value }); newRuleDomain.value = ''; newRuleTenantId.value = null; await loadRules(); } catch(e) { ruleError.value = e.message; }
      saving.value = false;
    }

    async function deleteRule(id, domain) {
      const ok = await showConfirm({ title: 'Delete Rule', message: 'Delete domain rule for "' + domain + '"?' });
      if (!ok) return; saving.value = true; ruleError.value = null;
      try { await api.del('/admin/domain-rules/' + id); await loadRules(); } catch(e) { ruleError.value = e.message; }
      saving.value = false;
    }

    // ── Credit Packs ──
    const creditPacks = ref([]);
    const packForm = reactive({ name: '', credits: null, price_paise: null });
    const packError = ref(null);

    async function loadPacks() { try { const d = await api.get('/admin/credit-packs'); creditPacks.value = d.items || d; } catch(e) { console.error(e); } }
    async function createPack() { packError.value = null; saving.value = true; try { await api.post('/admin/credit-packs', { ...packForm }); Object.assign(packForm, { name: '', credits: null, price_paise: null }); await loadPacks(); } catch(e) { packError.value = e.message; } saving.value = false; }
    async function togglePack(p) { packError.value = null; saving.value = true; try { await api.put('/admin/credit-packs/' + p.id, { is_active: !p.is_active }); await loadPacks(); } catch(e) { packError.value = e.message; } saving.value = false; }
    async function deletePack(id) { packError.value = null; const ok = await showConfirm({ title: 'Delete Pack', message: 'Delete this credit pack?' }); if (!ok) return; saving.value = true; try { await api.del('/admin/credit-packs/' + id); await loadPacks(); } catch(e) { packError.value = e.message; } saving.value = false; }

    // ── Time Pass Tiers ──
    const timePassTiers = ref([]);
    const tierForm = reactive({ name: '', duration_days: null, price_paise: null });
    const tierError = ref(null);

    async function loadTiers() { try { const d = await api.get('/admin/time-pass-tiers'); timePassTiers.value = d.items || d; } catch(e) { console.error(e); } }
    async function createTier() { tierError.value = null; saving.value = true; try { await api.post('/admin/time-pass-tiers', { ...tierForm }); Object.assign(tierForm, { name: '', duration_days: null, price_paise: null }); await loadTiers(); } catch(e) { tierError.value = e.message; } saving.value = false; }
    async function toggleTier(t) { tierError.value = null; saving.value = true; try { await api.put('/admin/time-pass-tiers/' + t.id, { is_active: !t.is_active }); await loadTiers(); } catch(e) { tierError.value = e.message; } saving.value = false; }
    async function deleteTier(id) { tierError.value = null; const ok = await showConfirm({ title: 'Delete Tier', message: 'Delete this time pass tier?' }); if (!ok) return; saving.value = true; try { await api.del('/admin/time-pass-tiers/' + id); await loadTiers(); } catch(e) { tierError.value = e.message; } saving.value = false; }

    // ── Promo Codes ──
    const promoCodes = ref([]);
    const promoForm = reactive({ code: '', type: 'CREDITS', value: null, max_redemptions: 0 });
    const promoError = ref(null);

    async function loadPromos() { try { const d = await api.get('/admin/promo-codes'); promoCodes.value = d.items || d; } catch(e) { console.error(e); } }
    async function createPromo() { promoError.value = null; saving.value = true; try { await api.post('/admin/promo-codes', { ...promoForm }); Object.assign(promoForm, { code: '', type: 'CREDITS', value: null, max_redemptions: 0 }); await loadPromos(); } catch(e) { promoError.value = e.message; } saving.value = false; }
    async function togglePromo(p) { promoError.value = null; saving.value = true; try { await api.put('/admin/promo-codes/' + p.id, { is_active: !p.is_active }); await loadPromos(); } catch(e) { promoError.value = e.message; } saving.value = false; }
    async function deletePromo(id) { promoError.value = null; const ok = await showConfirm({ title: 'Delete Promo', message: 'Delete this promo code?' }); if (!ok) return; saving.value = true; try { await api.del('/admin/promo-codes/' + id); await loadPromos(); } catch(e) { promoError.value = e.message; } saving.value = false; }

    // ── Transactions ──
    const adminTxns = ref([]);
    const txnSearch = ref('');
    const adminTxnPage = ref(1);
    const adminTxnPages = ref(1);

    async function loadTransactions(page = 1) {
      try {
        const d = await api.get(`/admin/transactions?page=${page}&size=50&search=${encodeURIComponent(txnSearch.value)}`);
        adminTxns.value = d.items; adminTxnPage.value = d.page; adminTxnPages.value = d.pages;
      } catch(e) { userError.value = 'Failed to load transactions: ' + e.message; }
    }

    // ── Tab switching ──
    function switchTab(key) {
      activeTab.value = key;
      tenantError.value = null; ruleError.value = null; userError.value = null;
      renameError.value = null; editing.value = null; packError.value = null;
      tierError.value = null; promoError.value = null;
      expandedUser.value = null; userDetail.value = null;
      if (key === 'overview') loadOverview();
      else if (key === 'users') loadUsers(1);
      else if (key === 'catalog') { loadPacks(); loadTiers(); loadPromos(); }
      else if (key === 'transactions') loadTransactions(1);
      else if (key === 'settings') { loadTenants(); loadRules(); }
    }

    onMounted(async () => {
      loading.value = true;
      await Promise.all([loadOverview(), loadTenants(), loadUsers(1)]);
      loading.value = false;
    });

    return {
      tabs, activeTab, loading, saving, overview, formatDate,
      formatTokens, barHeight, statusColor, funnelSteps, trendSeries, trendDayLabels,
      users, userSearch, userPage, userPages, userError, expandedUser, userDetail, userDetailLoading, userDetailPage,
      inlineGrantAmount, inlineGrantMsg, inlineGrantError, loadUsers, toggleUserDetail, loadUserDetail, inlineGrant, jumpToUser,
      tenants, newTenantName, tenantError, editing, editName, renameError, createTenant, renameTenant, deleteTenant, assignTenant,
      rules, newRuleDomain, newRuleTenantId, ruleError, createRule, deleteRule,
      creditPacks, packForm, packError, createPack, togglePack, deletePack,
      timePassTiers, tierForm, tierError, createTier, toggleTier, deleteTier,
      promoCodes, promoForm, promoError, createPromo, togglePromo, deletePromo,
      adminTxns, txnSearch, adminTxnPage, adminTxnPages, loadTransactions,
      switchTab,
    };
  },
};

// ================================================================
// PAGES — AI Settings
// ================================================================
const AISettingsPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div class="text-sm font-mono">
            <span class="font-bold text-white">AI SETTINGS</span>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Configure your AI provider and model</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div class="max-w-lg mx-auto">
          <div class="card p-6">
            <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">Provider</label>
            <select v-model="form.provider" @change="onProviderChange" class="input-field w-full mb-4">
              <option value="PLATFORM_GEMINI">Platform Default (Gemini)</option>
              <option value="GEMINI">Google Gemini (Own Key)</option>
              <option value="OPENAI">OpenAI</option>
              <option value="ANTHROPIC">Anthropic</option>
              <option value="CUSTOM_OPENAI_COMPATIBLE">Custom (OpenAI-compatible)</option>
            </select>

            <div v-if="form.provider !== 'PLATFORM_GEMINI'" class="mb-4">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">API Key</label>
              <input v-model="form.api_key" type="password" :placeholder="keyHint || 'Enter your API key'" class="input-field w-full">
            </div>

            <div v-if="form.provider === 'CUSTOM_OPENAI_COMPATIBLE'" class="mb-4">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">API Host</label>
              <input v-model="form.api_host" type="text" placeholder="https://api.groq.com/openai/v1" class="input-field w-full">
            </div>

            <button @click="fetchModels" :disabled="fetchingModels" class="btn-secondary w-full mb-4">
              {{ fetchingModels ? 'Fetching...' : 'Fetch Available Models' }}
            </button>

            <div class="mb-6">
              <label class="block text-[10px] font-mono font-bold tracking-widest uppercase mb-2" style="color:var(--text-dim)">Model</label>
              <select v-if="models.length > 0" v-model="form.model_id" class="input-field w-full">
                <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name || m.id }}</option>
              </select>
              <input v-else v-model="form.model_id" type="text" placeholder="Enter model ID" class="input-field w-full">
            </div>

            <div class="flex gap-3">
              <button @click="save" :disabled="saving" class="btn-primary flex-1 justify-center">
                {{ saving ? 'Saving...' : 'Save' }}
              </button>
              <button @click="reset" :disabled="saving" class="btn-secondary">
                Reset to Default
              </button>
            </div>

            <p v-if="error" class="text-red-400 text-xs mt-3 font-mono">{{ error }}</p>
            <p v-if="success" class="text-green-400 text-xs mt-3 font-mono">{{ success }}</p>

            <p class="text-[10px] font-mono mt-4" style="color:var(--text-dim)">
              Your API key is encrypted at rest. Platform default uses our Gemini API.
            </p>
          </div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const form = ref({
      provider: 'PLATFORM_GEMINI',
      api_key: '',
      api_host: '',
      model_id: '',
    });
    const models = ref([]);
    const fetchingModels = ref(false);
    const saving = ref(false);
    const error = ref('');
    const success = ref('');
    const keyHint = ref('');

    const load = async () => {
      try {
        const resp = await api.get('/settings/ai/');
        if (resp) {
          form.value.provider = resp.provider;
          form.value.model_id = resp.model_id;
          form.value.api_host = resp.api_host || '';
          form.value.api_key = '';
          keyHint.value = resp.key_hint || '';
        }
      } catch {}
    };

    const onProviderChange = () => {
      form.value.api_key = '';
      form.value.api_host = '';
      form.value.model_id = '';
      models.value = [];
      keyHint.value = '';
      error.value = '';
      success.value = '';
    };

    const fetchModels = async () => {
      fetchingModels.value = true;
      error.value = '';
      try {
        const body = {
          provider: form.value.provider,
          api_key: form.value.api_key || undefined,
          api_host: form.value.api_host || undefined,
        };
        const resp = await api.post('/settings/ai/models', body);
        models.value = resp || [];
        if (models.value.length === 0) {
          error.value = 'No models returned. You can type a model ID manually.';
        }
      } catch (e) {
        error.value = e.message || 'Failed to fetch models';
        models.value = [];
      } finally {
        fetchingModels.value = false;
      }
    };

    const save = async () => {
      saving.value = true;
      error.value = '';
      success.value = '';
      try {
        const body = {
          provider: form.value.provider,
          model_id: form.value.model_id,
          api_key: form.value.api_key || undefined,
          api_host: form.value.api_host || undefined,
        };
        await api.put('/settings/ai/', body);
        success.value = 'Settings saved!';
        form.value.api_key = '';
        await load();
      } catch (e) {
        error.value = e.message || 'Failed to save';
      } finally {
        saving.value = false;
      }
    };

    const reset = async () => {
      saving.value = true;
      error.value = '';
      success.value = '';
      try {
        await api.del('/settings/ai/');
        form.value = { provider: 'PLATFORM_GEMINI', api_key: '', api_host: '', model_id: '' };
        models.value = [];
        keyHint.value = '';
        success.value = 'Reset to platform default.';
      } catch (e) {
        error.value = e.message || 'Failed to reset';
      } finally {
        saving.value = false;
      }
    };

    onMounted(load);

    return { form, models, fetchingModels, saving, error, success, keyHint, onProviderChange, fetchModels, save, reset };
  },
};

const AppLayout = {
  components: { AppSidebar, ConfirmModal, PaywallModal },
  template: `
    <div class="flex h-full overflow-hidden relative z-10">
      <div class="sidebar-backdrop" :class="{ visible: sidebarOpen }" @click="closeSidebar"></div>
      <aside class="sidebar" :class="{ 'sidebar-open': sidebarOpen }">
        <AppSidebar />
      </aside>
      <div class="flex-1 flex flex-col overflow-hidden">
        <router-view v-slot="{ Component, route }">
          <transition name="page" mode="out-in">
            <component :is="Component" :key="route.fullPath" />
          </transition>
        </router-view>
      </div>
      <ConfirmModal />
      <PaywallModal />
    </div>
  `,
  setup() {
    const sidebarOpen = ref(false);
    const route = useRoute();
    const toggleSidebar = () => { sidebarOpen.value = !sidebarOpen.value; };
    const closeSidebar = () => { sidebarOpen.value = false; };
    // Auto-close sidebar on route change
    watch(() => route.fullPath, () => { sidebarOpen.value = false; });
    // Provide to child components (TopHeader injects this)
    provide('toggleSidebar', toggleSidebar);
    return { sidebarOpen, closeSidebar };
  },
};

// ================================================================
// PAGES — Resume Roast
// ================================================================
const RoastPage = {
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div class="min-w-0">
            <div class="flex items-center gap-2 text-sm font-mono min-w-0">
              <span v-if="phase === 'upload'" class="font-bold text-white">RESUME ROAST</span>
              <template v-else>
                <a v-if="phase !== 'processing'" @click.prevent="goBack" href="#" class="font-bold hover:text-white transition flex-shrink-0" style="color:var(--text-dim)">RESUME ROAST</a>
                <span v-else class="font-bold flex-shrink-0" style="color:var(--text-dim)">RESUME ROAST</span>
                <span style="color:var(--text-dim)" class="flex-shrink-0">/</span>
                <span class="font-bold text-white truncate">{{ phase === 'processing' ? 'Processing' : phase === 'result' ? 'Result' : 'Failed' }}</span>
              </template>
            </div>
            <p class="text-[10px] font-mono mt-0.5 hidden md:block" style="color:var(--text-dim)">Upload a resume and get brutally honest feedback</p>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div :class="phase === 'result' ? 'max-w-2xl' : 'max-w-xl'" class="mx-auto transition-all duration-300">

          <!-- ====== UPLOAD ZONE ====== -->
          <div v-if="phase === 'upload'" class="fade-slide-in">
            <div class="upload-zone mb-6"
                 :class="{ dragover: isDragging }"
                 @dragover.prevent="isDragging = true"
                 @dragleave="isDragging = false"
                 @drop.prevent="handleDrop"
                 @click="$refs.fileInput.click()">
              <input type="file" ref="fileInput" accept=".pdf" class="hidden" @change="handleFileSelect">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" class="mx-auto mb-4" style="opacity:0.7;">
                <path d="M12 22c-4 0-8-2-8-8 0-4 2-6 4-8 0 3 1.5 4 3 4-1-3 1-7 5-10 0 4 2 6 3 7 1.5 1.5 2 3 2 5 0 6-4 10-9 10z" fill="none" stroke="var(--orange)" stroke-width="1.5"/>
              </svg>
              <p class="text-white font-semibold mb-1" v-if="!selectedFile">Drop your PDF here or click to browse</p>
              <p class="text-white font-semibold mb-1" v-else>{{ selectedFile.name }}</p>
              <p class="text-xs" style="color:var(--text-dim)">PDF files only, max 5MB</p>
            </div>
            <button class="btn-primary w-full justify-center" @click="doUpload"
                    :disabled="!selectedFile">
              ROAST MY RESUME
            </button>
            <p v-if="error" class="text-red-400 text-xs mt-3 text-center font-mono">{{ error }}</p>

            <!-- Previous roasts -->
            <div v-if="store.roasts.length > 0" class="mt-8">
              <div class="section-label mb-3">Previous Roasts</div>
              <div class="space-y-2">
                <div v-for="r in store.roasts" :key="r.id"
                     class="widget-card transition p-4 cursor-pointer hover:border-white/10"
                     @click="viewRoast(r)">
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3 min-w-0">
                      <div class="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                           :style="{ background: r.roast_data ? scoreColor(r.roast_data.score, 0.15) : 'rgba(255,255,255,0.05)' }">
                        <span v-if="r.roast_data" class="font-mono font-bold text-xs" :style="{ color: scoreColor(r.roast_data.score) }">{{ r.roast_data.score }}</span>
                        <span v-else class="text-xs" style="color:var(--text-dim)">-</span>
                      </div>
                      <div class="min-w-0">
                        <p class="text-white text-sm font-semibold truncate">{{ roastListLabel(r) }}</p>
                        <p class="text-[10px] font-mono" style="color:var(--text-dim)">{{ fmtDate(r.created_at) }}</p>
                      </div>
                    </div>
                    <span :class="badgeClass(r.status)" class="badge flex-shrink-0">{{ r.status }}</span>
                  </div>
                </div>
              </div>
              <PaginationControls :page="store.page" :totalPages="store.totalPages" @go="store.goToPage" />
            </div>
          </div>

          <!-- ====== PROCESSING ====== -->
          <div v-if="phase === 'processing'" class="fade-slide-in">
            <div class="widget-card mb-4">
              <div class="flex items-center gap-3 mb-3">
                <div class="spinner" style="width:20px;height:20px;border-width:2px;"></div>
                <p class="text-white font-semibold text-sm font-mono">{{ phaseLabel }}</p>
              </div>
              <div class="progress-track" style="max-width:100%;">
                <div class="progress-fill transition-all duration-700" :style="{ width: progress + '%' }"></div>
              </div>
              <div class="mt-3 space-y-1">
                <div v-for="(line, i) in statusLines" :key="i" class="boot-line visible" :class="{ ok: line.ok }">
                  {{ line.text }}
                </div>
              </div>
            </div>
            <!-- OCR text cycling -->
            <div v-if="ocrLines.length > 0 && !ocrDone" class="widget-card" style="border-left: 2px solid var(--teal);">
              <div class="flex items-center gap-2 mb-3">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                <span class="text-[10px] font-mono font-bold tracking-widest" style="color:var(--teal)">READING RESUME</span>
              </div>
              <div class="overflow-hidden relative flex flex-col justify-end" style="height:180px;">
                <div class="space-y-0.5">
                  <div v-for="(line, i) in visibleOcrLines" :key="i"
                       class="font-mono text-[11px] leading-relaxed transition-opacity duration-300"
                       :style="{ color: i === visibleOcrLines.length - 1 ? 'var(--text-primary)' : 'var(--text-dim)', opacity: i < visibleOcrLines.length - 3 ? 0.3 : (i < visibleOcrLines.length - 1 ? 0.6 : 1) }">
                    <span v-if="i === visibleOcrLines.length - 1" style="color:var(--teal);">&gt; </span>{{ line }}
                  </div>
                </div>
              </div>
            </div>
            <!-- AI judging -->
            <div v-if="ocrDone" class="widget-card fade-slide-in" style="border-left: 2px solid var(--orange);">
              <div class="flex items-center gap-3">
                <div class="spinner" style="width:18px;height:18px;border-width:2px;border-color:rgba(251,65,0,0.2);border-top-color:var(--orange);"></div>
                <div>
                  <p class="text-sm font-mono font-semibold" style="color:var(--orange)">AI IS JUDGING YOU</p>
                  <p class="text-[10px] font-mono mt-0.5" style="color:var(--text-dim)">Generating your roast... this might hurt</p>
                </div>
              </div>
            </div>
          </div>

          <!-- ====== RESULT ====== -->
          <div v-if="phase === 'result' && roastData" class="fade-slide-in space-y-4">
            <!-- Share link bar (top) -->
            <div v-if="currentShareId" class="flex items-center justify-end">
              <button class="flex items-center gap-2 py-1.5 px-3 rounded-md font-mono text-[10px] font-bold tracking-wider transition"
                      style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); color:var(--text-dim);"
                      @click="shareRoast"
                      @mouseenter="$event.currentTarget.style.borderColor='var(--teal)'; $event.currentTarget.style.color='var(--teal)'"
                      @mouseleave="$event.currentTarget.style.borderColor='rgba(255,255,255,0.08)'; $event.currentTarget.style.color='var(--text-dim)'">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
                  <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
                </svg>
                {{ shareToast || 'SHARE LINK' }}
              </button>
            </div>

            <!-- Score card (always visible above tabs) -->
            <div class="widget-card" style="border-left: 3px solid var(--orange);">
              <div class="flex items-start gap-4">
                <div class="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0"
                     :style="{ background: scoreColor(roastData.score, 0.12) }">
                  <span class="font-mono font-black text-2xl" :style="{ color: scoreColor(roastData.score) }">{{ roastData.score }}</span>
                </div>
                <div class="min-w-0">
                  <p class="text-white font-bold text-lg leading-snug">{{ roastData.headline }}</p>
                  <p class="text-[10px] font-mono mt-1" :style="{ color: scoreColor(roastData.score) }">{{ scoreLabel(roastData.score) }}</p>
                </div>
              </div>
            </div>

            <!-- Tab bar -->
            <div class="flex" style="border-bottom:1px solid rgba(255,255,255,0.1);">
              <button @click="roastTab = 'roast'"
                      class="font-mono font-bold text-xs tracking-wider transition-all duration-200"
                      :style="{ padding:'12px 24px', background:'transparent', border:'none', borderBottom: roastTab === 'roast' ? '2px solid #f97316' : '2px solid transparent', color: roastTab === 'roast' ? '#f97316' : 'rgba(255,255,255,0.5)', cursor:'pointer', textTransform:'uppercase', letterSpacing:'1px' }">
                ROAST
              </button>
              <button @click="roastTab = 'check'"
                      class="font-mono font-bold text-xs tracking-wider transition-all duration-200"
                      :style="{ padding:'12px 24px', background:'transparent', border:'none', borderBottom: roastTab === 'check' ? '2px solid #f97316' : '2px solid transparent', color: roastTab === 'check' ? '#f97316' : 'rgba(255,255,255,0.5)', cursor:'pointer', textTransform:'uppercase', letterSpacing:'1px' }">
                REALITY CHECK
              </button>
            </div>

            <!-- ROAST tab -->
            <div v-if="roastTab === 'roast'" class="space-y-4">
              <!-- Roast points -->
              <div class="widget-card">
                <div class="section-label mb-3">THE ROAST</div>
                <div class="space-y-3">
                  <div v-for="(point, i) in roastData.roast_points" :key="i"
                       class="flex items-start gap-3 p-3 rounded-lg" style="background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.04);">
                    <span class="text-xl flex-shrink-0 mt-0.5">{{ point.emoji }}</span>
                    <p class="text-sm text-white leading-relaxed">{{ point.text }}</p>
                  </div>
                </div>
              </div>

              <!-- Actual feedback -->
              <div class="widget-card" style="border-left: 3px solid var(--teal);">
                <div class="section-label mb-3" style="color:var(--teal)">ACTUAL FEEDBACK</div>
                <p class="text-sm leading-relaxed" style="color:var(--text-secondary)">{{ roastData.actual_feedback }}</p>
              </div>

              <!-- Verdict -->
              <div class="widget-card text-center py-6">
                <p class="text-[10px] font-mono font-bold tracking-widest mb-2" style="color:var(--text-dim)">THE VERDICT</p>
                <p class="text-white font-bold text-lg">{{ roastData.verdict }}</p>
              </div>
            </div>

            <!-- REALITY CHECK tab -->
            <div v-if="roastTab === 'check'" class="space-y-4">
              <!-- ATS Readiness Checklist -->
              <div v-if="roastData.ats_checklist && roastData.ats_checklist.length > 0">
                <!-- Score header card -->
                <div class="widget-card mb-4">
                  <div class="flex items-center justify-between mb-4">
                    <div class="section-label mb-0">ATS Readiness</div>
                    <span class="font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded"
                          :style="{ color: atsPassStyle.color, background: atsPassStyle.bg }">
                      {{ atsPassCount }} CRITERIA MET
                    </span>
                  </div>
                  <!-- Progress bar -->
                  <div class="w-full rounded-full overflow-hidden" style="height:6px;background:rgba(255,255,255,0.06);">
                    <div class="h-full rounded-full transition-all duration-500"
                         :style="{ width: (roastData.ats_checklist.filter(c => c.passed).length / roastData.ats_checklist.length * 100) + '%', background: atsPassStyle.color }"></div>
                  </div>
                </div>

                <!-- Individual check items — unified list -->
                <div class="space-y-2">
                  <div v-for="(check, i) in roastData.ats_checklist" :key="'ats-'+i"
                       class="widget-card flex items-start gap-3"
                       :style="{ borderLeft: check.passed ? '3px solid #22c55e' : '3px solid #fbbf24', padding: '14px 16px' }">
                    <div class="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                         :style="{ background: check.passed ? 'rgba(34,197,94,0.15)' : 'rgba(251,191,36,0.15)' }">
                      <span v-if="check.passed" class="text-xs font-bold" style="color:#22c55e;">&#x2713;</span>
                      <span v-else class="text-xs font-bold" style="color:#fbbf24;">!</span>
                    </div>
                    <div class="min-w-0 flex-1">
                      <p class="text-white text-sm font-semibold">{{ check.label }}</p>
                      <p v-if="!check.passed && check.detail" class="text-xs mt-1 leading-relaxed" style="color:var(--text-dim)">{{ check.detail }}</p>
                      <p v-if="check.passed && getAtsDescription(check.label)" class="text-xs mt-1 leading-relaxed" style="color:var(--text-dim)">{{ getAtsDescription(check.label) }}</p>
                    </div>
                  </div>
                </div>

                <!-- Educational note -->
                <div class="mt-4 p-4 rounded-lg" style="background:rgba(1,169,219,0.06);border:1px solid rgba(1,169,219,0.1);">
                  <p class="text-xs leading-relaxed" style="color:var(--teal)"><span class="font-semibold">There's no such thing as an "ATS score."</span> ATS software doesn't rate your resume out of 100 — that number you see on other sites is made up. What actually happens is simple: the system checks if your resume meets basic formatting criteria (that's what we show above) and then pattern-matches your content against the job description's keywords, skills, and qualifications. It's pass or fail, not a score.</p>
                  <p class="text-xs leading-relaxed mt-2" style="color:var(--teal)">The formatting checks above are real and useful — but they're only half the battle. The other half is keyword alignment with a specific role. That's exactly what our resume tailoring does: it restructures your resume to match a real job description, so you actually get past the filter.</p>
                </div>
              </div>

              <!-- CTA: Roast to Profile conversion -->
              <div v-if="showProfileCta" class="widget-card" style="border-left:3px solid var(--teal);background:rgba(1,169,219,0.06);">
                <div class="flex flex-col sm:flex-row items-start sm:items-center gap-3 p-1">
                  <div class="flex-1 min-w-0">
                    <p class="text-white font-semibold text-sm">Now fix it with AI</p>
                    <p class="text-xs mt-0.5" style="color:var(--text-dim)">Upload your resume to create a structured profile, then generate tailored resumes for any job.</p>
                  </div>
                  <router-link to="/profiles/new" class="btn-primary text-xs flex-shrink-0 whitespace-nowrap">UPLOAD RESUME</router-link>
                </div>
              </div>
            </div>

            <!-- Roast another (always visible) -->
            <button class="btn-primary w-full justify-center" @click="reset" style="background:rgba(255,255,255,0.06);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);">
              ROAST ANOTHER
            </button>
          </div>

          <!-- ====== FAILED ====== -->
          <div v-if="phase === 'failed'" class="widget-card text-center py-12 fade-slide-in">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5" class="mx-auto mb-4">
              <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <p class="text-white font-semibold mb-1">Roast failed</p>
            <p class="text-xs mb-6" style="color:var(--text-dim)">Something went wrong. Please try again.</p>
            <button class="btn-primary" @click="reset">TRY AGAIN</button>
          </div>

        </div>
      </div>
    </div>
  `,
  setup() {
    const store = useRoastStore();
    const phase = ref('upload'); // upload | processing | result | failed
    const selectedFile = ref(null);
    const isDragging = ref(false);
    const error = ref(null);
    const roastData = ref(null);
    const currentShareId = ref(null);
    const shareToast = ref(null);
    const roastTab = ref('roast');
    // Always show profile CTA on roast result for logged-in users
    const showProfileCta = computed(() => phase.value === 'result');

    // ATS checklist computed properties
    const atsPassCount = computed(() => {
      const checks = roastData.value?.ats_checklist;
      if (!checks || !checks.length) return '0/0';
      return `${checks.filter(c => c.passed).length}/${checks.length}`;
    });
    const atsDescriptions = {
      'machine readability': 'ATS software can extract text cleanly from your PDF without garbled characters or missing sections.',
      'standard section headers': 'Uses recognizable headers like Experience, Education, Skills that ATS parsers expect.',
      'single-column layout': 'Clean single-column format that won\'t break when parsed by automated systems.',
      'single column layout': 'Clean single-column format that won\'t break when parsed by automated systems.',
      'contact info in body': 'Name, email, and phone are in the main body where ATS can find them, not hidden in headers/footers.',
      'contact information': 'Name, email, and phone are in the main body where ATS can find them, not hidden in headers/footers.',
      'no graphics dependency': 'All critical info is in parseable text — no icons, charts, or text baked into images.',
      'avoids tables/graphics': 'All critical info is in parseable text — no icons, charts, or text baked into images.',
      'skills section': 'Dedicated skills section listing technical and relevant skills for ATS keyword matching.',
      'keyword optimization': 'Key skills and technologies are explicitly listed for ATS keyword matching.',
      'consistent date formatting': 'Dates use the same format throughout, making them parseable by automated systems.',
      'consistent date formats': 'Dates use the same format throughout, making them parseable by automated systems.',
      'quantified achievements': 'Bullet points include numbers and metrics that demonstrate measurable impact.',
      'action verbs': 'Bullet points lead with strong action verbs that convey impact and ownership.',
      'education relevance': 'Education section is present with degree, institution, and relevant details.',
    };
    const getAtsDescription = (label) => atsDescriptions[label.toLowerCase()] || null;

    const atsPassStyle = computed(() => {
      const checks = roastData.value?.ats_checklist;
      if (!checks || !checks.length) return { color: 'var(--text-dim)', bg: 'rgba(255,255,255,0.05)' };
      const passed = checks.filter(c => c.passed).length;
      const ratio = passed / checks.length;
      // Smooth red→yellow→green gradient via HSL (0°=red, 45°=yellow, 142°=green)
      const hue = Math.round(ratio * 142);
      const sat = ratio < 0.5 ? 80 : 70;
      const lum = ratio < 0.5 ? 50 : 45 + ratio * 10;
      const color = `hsl(${hue}, ${sat}%, ${Math.round(lum)}%)`;
      const bg = `hsla(${hue}, ${sat}%, ${Math.round(lum)}%, 0.12)`;
      return { color, bg };
    });
    const progress = ref(0);
    const statusLines = ref([]);
    const phaseLabel = ref('Uploading...');
    let unmounted = false;

    // OCR text cycling (reused from ProfileUploadPage)
    const ocrLines = ref([]);
    const visibleOcrLines = ref([]);
    const ocrDone = ref(false);
    let ocrTimer = null;
    let progressTimerRef = null;
    let ocrDoneCheckTimer = null;
    const MAX_VISIBLE_LINES = 9;

    const startOcrCycling = (text) => {
      const lines = text.split(/\n/).map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length === 0) return;
      ocrLines.value = lines;
      ocrDone.value = false;
      let idx = 0;
      visibleOcrLines.value = [lines[0]];
      idx = 1;
      const delay = Math.max(150, Math.min(350, 20000 / lines.length));
      ocrTimer = setInterval(() => {
        if (idx < lines.length) {
          const next = [...visibleOcrLines.value, lines[idx]];
          visibleOcrLines.value = next.length > MAX_VISIBLE_LINES ? next.slice(-MAX_VISIBLE_LINES) : next;
          idx++;
        } else {
          clearInterval(ocrTimer);
          ocrTimer = null;
          ocrDone.value = true;
        }
      }, delay);
    };

    const waitForOcrDone = () => new Promise(resolve => {
      if (ocrDone.value) { resolve(); return; }
      ocrDoneCheckTimer = setInterval(() => {
        if (ocrDone.value) { clearInterval(ocrDoneCheckTimer); ocrDoneCheckTimer = null; resolve(); }
      }, 200);
    });

    const stopOcrCycling = () => {
      if (ocrTimer) { clearInterval(ocrTimer); ocrTimer = null; }
      if (progressTimerRef) { clearInterval(progressTimerRef); progressTimerRef = null; }
      // Force ocrDone so waitForOcrDone promise resolves (prevents hanging on unmount)
      ocrDone.value = true;
      if (ocrDoneCheckTimer) { clearInterval(ocrDoneCheckTimer); ocrDoneCheckTimer = null; }
    };

    const handleDrop = (e) => {
      isDragging.value = false;
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (file.type !== 'application/pdf') {
        error.value = 'Only PDF files are accepted.';
        return;
      }
      error.value = null;
      selectedFile.value = file;
    };
    const handleFileSelect = (e) => {
      const file = e.target.files[0];
      if (file) { error.value = null; selectedFile.value = file; }
    };

    const doUpload = async () => {
      if (!selectedFile.value) return;
      if (selectedFile.value.type !== 'application/pdf') {
        error.value = 'Only PDF files are accepted.';
        return;
      }
      if (selectedFile.value.size > 5 * 1024 * 1024) {
        error.value = 'File size must be under 5MB. Try exporting a smaller version without embedded images.';
        return;
      }

      phase.value = 'processing';
      error.value = null;
      progress.value = 10;
      phaseLabel.value = 'Uploading PDF...';
      statusLines.value = [{ text: '[SYS] Uploading PDF to server...', ok: false }];

      try {
        const result = await store.uploadRoast(selectedFile.value);
        if (unmounted) return;
        const isCached = result.cached && result.status === 'READY';

        // Always show processing phase — even for cached results
        progress.value = 30;
        phaseLabel.value = 'Reading your resume...';
        statusLines.value = [
          { text: '[SYS] Upload complete', ok: true },
          { text: '[OCR] Text extracted from PDF', ok: true },
          { text: '[AI] Generating roast...', ok: false },
        ];

        // Start OCR cycling (always, even for cached)
        if (result.extracted_text && result.extracted_text.trim()) {
          startOcrCycling(result.extracted_text);
        }
        if (ocrLines.value.length === 0) {
          ocrDone.value = true;
        }

        progress.value = 40;
        progressTimerRef = setInterval(() => {
          if (progress.value < 85) progress.value += 2;
        }, 1500);

        let status;
        if (isCached) {
          // Cached: cycle through text, show "AI IS JUDGING YOU", then deliberate delay
          await waitForOcrDone();
          if (unmounted) return;
          await new Promise(r => setTimeout(r, 1500));
          if (unmounted) return;
          status = 'READY';
        } else {
          // Non-cached: wait for both AI completion AND OCR cycling to finish
          const [pollResult] = await Promise.all([
            store.pollStatus(result.roast_id),
            waitForOcrDone(),
          ]);
          if (unmounted) return;
          status = pollResult;
        }

        stopOcrCycling();
        progress.value = 100;

        if (status === 'READY') {
          phaseLabel.value = 'Roast ready!';
          statusLines.value.push({ text: '[SYS] Your roast is served!', ok: true });
          await new Promise(r => setTimeout(r, 400));
          if (unmounted) return;
          await store.fetchRoast(result.roast_id);
          if (unmounted) return;
          roastData.value = store.current?.roast_data || null;
          currentShareId.value = store.current?.share_id || result.share_id || null;
          roastTab.value = 'roast';
          phase.value = roastData.value ? 'result' : 'failed';
        } else {
          phase.value = 'failed';
        }
      } catch (e) {
        stopOcrCycling();
        if (unmounted) return;
        error.value = e.message;
        phase.value = 'upload';
      }
    };

    const viewRoast = async (r) => {
      if (r.status === 'READY' && r.roast_data) {
        roastData.value = r.roast_data;
        currentShareId.value = r.share_id || null;
        roastTab.value = 'roast';
        phase.value = 'result';
      }
    };

    const reset = () => {
      phase.value = 'upload';
      selectedFile.value = null;
      error.value = null;
      roastData.value = null;
      currentShareId.value = null;
      shareToast.value = null;
      roastTab.value = 'roast';
      progress.value = 0;
      statusLines.value = [];
      ocrLines.value = [];
      visibleOcrLines.value = [];
      ocrDone.value = false;
      stopOcrCycling();
      store.fetchRoasts(true);
    };

    const goBack = () => {
      if (phase.value === 'processing') return; // don't interrupt processing
      reset();
    };

    const shareRoast = async () => {
      if (!currentShareId.value) return;
      const url = `${window.location.origin}/roast/${currentShareId.value}`;
      try {
        await navigator.clipboard.writeText(url);
      } catch {
        const ta = document.createElement('textarea');
        ta.value = url;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      shareToast.value = 'LINK COPIED!';
      setTimeout(() => { shareToast.value = null; }, 2000);
    };

    const scoreColor = (score, alpha) => {
      if (score >= 8) return alpha ? `rgba(34,197,94,${alpha})` : '#22c55e';
      if (score >= 5) return alpha ? `rgba(251,191,36,${alpha})` : '#fbbf24';
      return alpha ? `rgba(239,68,68,${alpha})` : '#ef4444';
    };

    const scoreLabel = (score) => {
      if (score >= 9) return 'EXCEPTIONAL';
      if (score >= 7) return 'SOLID';
      if (score >= 5) return 'MEDIOCRE';
      if (score >= 3) return 'ROUGH';
      return 'DUMPSTER FIRE';
    };

    const roastListLabel = (r) => {
      if (r.roast_data && r.roast_data.headline) return r.roast_data.headline;
      return 'Processing...';
    };

    const badgeClass = (status) => {
      const map = { PENDING: 'badge-pending', PROCESSING: 'badge-processing', READY: 'badge-ready', FAILED: 'badge-failed' };
      return map[status] || 'badge-pending';
    };

    const fmtDate = (iso) => formatDate(iso, { includeYear: true });

    onMounted(() => store.fetchRoasts(true));
    onUnmounted(() => { unmounted = true; stopOcrCycling(); });

    onBeforeRouteLeave(async () => {
      if (phase.value !== 'processing') return true;
      const leave = await showConfirm({
        title: 'Roast in Progress',
        message: 'Your resume roast is still being generated. It will continue in the background if you leave, but you won\'t see the live progress.',
        confirmLabel: 'LEAVE',
        variant: 'warning',
      });
      return leave;
    });

    return {
      store, phase, selectedFile, isDragging, error, roastData, currentShareId, shareToast, roastTab,
      progress, statusLines, phaseLabel, showProfileCta,
      atsPassCount, atsPassStyle, getAtsDescription,
      ocrLines, visibleOcrLines, ocrDone,
      handleDrop, handleFileSelect, doUpload, viewRoast, reset, goBack, shareRoast,
      scoreColor, scoreLabel, roastListLabel, badgeClass, fmtDate,
    };
  },
};

// ================================================================
// ROUTER
// ================================================================
// ================================================================
// PAGES — Credit History
// ================================================================
const CreditHistoryPage = {
  components: { PaginationControls },
  template: `
    <div>
      <TopHeader>
        <template #left>
          <div>
            <h1 class="text-sm font-bold text-white font-mono tracking-tight">CREDITS</h1>
            <p class="text-[10px] font-mono hidden md:block" style="color:var(--text-dim)">Manage your credits & purchases</p>
          </div>
        </template>
        <template #right>
          <div class="flex items-center gap-3">
            <div class="text-right hidden md:block">
              <span class="kpi-label">Available</span>
              <span class="font-mono font-bold text-base ml-2" style="color:var(--orange)">{{ creditStore.displayBalance }}</span>
            </div>
          </div>
        </template>
      </TopHeader>
      <div class="flex-1 overflow-y-auto p-4 md:p-6 page-scroll">
        <div class="max-w-4xl mx-auto">

          <!-- Balance hero -->
          <div class="rounded-xl p-5 mb-6" style="background:linear-gradient(135deg, rgba(251,65,0,0.08), rgba(1,169,219,0.06));border:1px solid rgba(251,65,0,0.12);">
            <div class="grid grid-cols-3 gap-4 text-center">
              <div>
                <p class="text-[10px] font-mono font-bold tracking-widest uppercase mb-1" style="color:var(--text-dim)">Purchased</p>
                <p class="font-mono font-bold text-3xl text-white">{{ creditStore.balance }}</p>
              </div>
              <div>
                <p class="text-[10px] font-mono font-bold tracking-widest uppercase mb-1" style="color:var(--text-dim)">Free Today</p>
                <p class="font-mono font-bold text-3xl" style="color:var(--teal)">{{ creditStore.dailyFreeRemaining }}<span class="text-lg" style="color:var(--text-dim)">/{{ creditStore.dailyFreeTotal }}</span></p>
              </div>
              <div>
                <p class="text-[10px] font-mono font-bold tracking-widest uppercase mb-1" style="color:var(--text-dim)">Time Pass</p>
                <p class="font-mono text-sm" style="color:var(--text-dim)">Coming Soon</p>
              </div>
            </div>
            <!-- Low credits nudge -->
            <div v-if="!creditStore.hasUnlimited && creditStore.balance === 0 && creditStore.dailyFreeRemaining === 0" class="mt-4 pt-3 border-t border-white/5 text-center">
              <p class="text-xs font-mono" style="color:var(--orange)">You're out of credits — grab a pack below to keep generating resumes</p>
            </div>
          </div>

          <!-- How credits work -->
          <div class="rounded-lg p-4 mb-6 flex items-start gap-3" style="background:rgba(1,169,219,0.04);border:1px solid rgba(1,169,219,0.1);">
            <span style="font-size:18px;line-height:1;">&#9889;</span>
            <div>
              <p class="text-xs font-semibold text-white mb-1">1 Credit = 1 Tailored Resume</p>
              <p class="text-[11px]" style="color:var(--text-dim)">{{ creditStore.dailyFreeTotal }} free daily. Buy more when you need them. Credits never expire.</p>
            </div>
          </div>

          <!-- Credit Packs -->
          <div v-if="packsLoading" class="flex justify-center py-8 mb-6"><div class="spinner"></div></div>
          <div v-else-if="creditStore.packs.length || creditStore.timePasses.length" class="mb-6">
            <div v-if="creditStore.packs.length" class="mb-6">
              <div class="section-label mb-3">Credit Packs</div>
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div v-for="(p, i) in creditStore.packs" :key="'pack-'+p.id"
                     class="relative rounded-xl p-5 flex flex-col transition-all hover:scale-[1.02]"
                     :style="bestPackIdx === i
                       ? 'background:rgba(251,65,0,0.06);border:2px solid rgba(251,65,0,0.35);'
                       : 'background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.06);'">
                  <!-- Best value badge -->
                  <div v-if="bestPackIdx === i" class="absolute -top-2.5 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full text-[9px] font-mono font-bold tracking-wider" style="background:var(--orange);color:white;">BEST VALUE</div>
                  <div class="text-white text-sm font-semibold mb-1">{{ p.name }}</div>
                  <div class="font-mono font-bold text-2xl text-white mb-1">{{ p.credits }} <span class="text-sm font-normal" style="color:var(--text-dim)">credits</span></div>
                  <div class="text-[10px] font-mono mb-4" style="color:var(--teal)">{{ formatPricePerCredit(p) }}/credit</div>
                  <button @click="buyPack('credit_pack', p.id, p.price_paise)" :disabled="buying || packsLoading"
                          :class="bestPackIdx === i ? 'btn-primary' : 'btn-ghost'"
                          class="w-full mt-auto text-sm">
                    {{ buying ? '...' : formatPrice(p.price_paise) }}
                  </button>
                </div>
              </div>
            </div>

            <div v-if="creditStore.timePasses.length" class="opacity-50">
              <div class="section-label mb-3">Time Passes — Unlimited Generations <span class="text-[9px] font-mono ml-1 px-1.5 py-0.5 rounded" style="background:rgba(1,169,219,0.15);color:var(--teal);">COMING SOON</span></div>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div v-for="t in creditStore.timePasses" :key="'tp-'+t.id"
                     class="rounded-xl p-5 flex flex-col"
                     style="background:rgba(1,169,219,0.04);border:1px solid rgba(1,169,219,0.2);">
                  <div class="text-white text-sm font-semibold mb-1">{{ t.name }}</div>
                  <div class="flex items-baseline gap-2 mb-1">
                    <span class="font-mono font-bold text-2xl text-white">{{ t.duration_days }}</span>
                    <span class="text-sm" style="color:var(--text-dim)">days</span>
                  </div>
                  <div class="text-[10px] font-mono mb-4" style="color:var(--teal)">Unlimited resume generations</div>
                  <div class="w-full mt-auto text-center text-[10px] font-mono py-2" style="color:var(--text-dim);">{{ formatPrice(t.price_paise) }}</div>
                </div>
              </div>
            </div>

            <p v-if="buyError" class="text-xs mt-3 font-mono" style="color:var(--red)">{{ buyError }}</p>
            <p v-if="buySuccess" class="text-xs mt-3 font-mono" style="color:var(--green)">{{ buySuccess }}</p>
          </div>

          <!-- Promo code — compact inline -->
          <div class="rounded-lg p-4 mb-6 flex items-center gap-3 flex-wrap" style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);">
            <span class="text-[10px] font-mono font-bold tracking-widest uppercase shrink-0" style="color:var(--text-dim)">Promo Code</span>
            <input v-model="promoCode" class="input-field input-mono flex-1" style="min-width:140px;" placeholder="Enter code" @keyup.enter="redeem" />
            <button @click="redeem" class="btn-ghost text-xs" :disabled="!promoCode.trim() || redeeming || packsLoading">{{ redeeming ? 'APPLYING...' : 'APPLY' }}</button>
            <p v-if="promoMsg" class="w-full text-xs font-mono mt-1" :style="{ color: promoIsError ? 'var(--red)' : 'var(--green)' }">{{ promoMsg }}</p>
          </div>

          <!-- Transactions — collapsible -->
          <div class="mb-6">
            <button @click="showHistory = !showHistory" class="flex items-center gap-2 mb-3 group cursor-pointer" style="background:none;border:none;padding:0;">
              <span class="section-label">Transaction History</span>
              <svg :style="{ transform: showHistory ? 'rotate(180deg)' : 'rotate(0)' }" style="transition:transform 0.2s;" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
              <span class="text-[10px] font-mono" style="color:var(--text-dim)">{{ creditStore.txnTotal }} entries</span>
            </button>
            <div v-if="showHistory" class="widget-card">
              <div v-if="creditStore.loading" class="flex items-center justify-center py-8"><div class="spinner"></div></div>
              <div v-else-if="creditStore.transactions.length === 0" class="empty-state"><p class="text-sm">No transactions yet</p></div>
              <div class="overflow-x-auto" v-else>
                <table class="data-table" style="min-width:500px;">
                  <thead><tr><th>Date</th><th>Type</th><th>Amount</th><th>Description</th></tr></thead>
                  <tbody>
                    <tr v-for="t in creditStore.transactions" :key="t.id">
                      <td class="font-mono text-[11px]" style="color:var(--text-dim)">{{ formatDate(t.created_at, { includeTime: true }) }}</td>
                      <td><span class="badge badge-processing">{{ t.type }}</span></td>
                      <td :style="{ color: t.amount >= 0 ? 'var(--green)' : 'var(--red)' }" class="font-mono font-bold">{{ t.amount >= 0 ? '+' : '' }}{{ t.amount }}</td>
                      <td class="text-[11px]" style="color:var(--text-dim)">{{ t.description || '-' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <PaginationControls v-if="creditStore.txnPages > 1" :page="creditStore.txnPage" :totalPages="creditStore.txnPages" @go="goPage" />
            </div>
          </div>

        </div>
      </div>
    </div>
  `,
  setup() {
    const creditStore = useCreditStore();
    const promoCode = ref('');
    const promoMsg = ref('');
    const promoIsError = ref(false);
    const redeeming = ref(false);
    const buying = ref(false);
    const buyError = ref(null);
    const buySuccess = ref(null);
    const packsLoading = ref(false);
    const showHistory = ref(false);
    let successTimer = null;

    // Best-value pack = lowest price per credit
    const bestPackIdx = computed(() => {
      if (!creditStore.packs.length) return -1;
      let best = 0;
      let bestRatio = Infinity;
      creditStore.packs.forEach((p, i) => {
        const ratio = p.price_paise / p.credits;
        if (ratio < bestRatio) { bestRatio = ratio; best = i; }
      });
      return creditStore.packs.length > 1 ? best : -1;
    });

    onMounted(async () => {
      creditStore.fetchBalance();
      creditStore.fetchHistory(1, '');
      packsLoading.value = true;
      await creditStore.fetchPacks();
      packsLoading.value = false;
    });

    onUnmounted(() => {
      if (successTimer) clearTimeout(successTimer);
    });

    function formatPrice(paise) { return '\u20B9' + (paise / 100).toFixed(0); }
    function formatPricePerCredit(pack) { return '\u20B9' + (pack.price_paise / pack.credits / 100).toFixed(1); }

    function goPage(n) { creditStore.fetchHistory(n, ''); }

    async function buyPack(itemType, itemId, amountPaise) {
      if (typeof Razorpay === 'undefined') {
        buyError.value = 'Payment gateway not loaded. Please refresh the page.';
        return;
      }
      buying.value = true;
      buyError.value = null;
      buySuccess.value = null;
      try {
        const order = await api.post('/payments/create-order', { item_type: itemType, item_id: itemId });
        const options = {
          key: order.razorpay_key_id,
          amount: order.amount_paise,
          currency: order.currency,
          order_id: order.order_id,
          name: 'ATS Beater',
          description: itemType === 'credit_pack' ? 'Credit Pack' : 'Time Pass',
          handler: async function(response) {
            try {
              await api.post('/payments/verify', {
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature,
              });
              await creditStore.fetchBalance();
              creditStore.fetchHistory(1, '');
              buySuccess.value = 'Purchase successful! Credits have been added.';
              if (successTimer) clearTimeout(successTimer);
              successTimer = setTimeout(() => { buySuccess.value = null; }, 5000);
            } catch (e) {
              buyError.value = 'Payment verification failed: ' + e.message;
            }
            buying.value = false;
          },
          modal: {
            ondismiss: function() { buying.value = false; },
          },
          theme: { color: '#FB4100' },
        };
        const rzp = new Razorpay(options);
        rzp.open();
      } catch (e) {
        buyError.value = e.message;
        buying.value = false;
      }
    }

    async function redeem() {
      if (!promoCode.value.trim() || redeeming.value) return;
      redeeming.value = true;
      promoMsg.value = '';
      promoIsError.value = false;
      try {
        const result = await creditStore.redeemPromo(promoCode.value);
        promoMsg.value = result.message;
        promoCode.value = '';
        creditStore.fetchHistory(1, '');
      } catch(e) {
        promoMsg.value = e.message;
        promoIsError.value = true;
      }
      redeeming.value = false;
    }

    return { creditStore, promoCode, promoMsg, promoIsError, redeeming, buying, buyError, buySuccess, packsLoading, showHistory, bestPackIdx, formatDate, formatPrice, formatPricePerCredit, goPage, redeem, buyPack };
  },
};

// ================================================================
// SHARED COMPONENT — Chat Panel (used by Job + Profile detail pages)
// ================================================================
const ChatPanel = {
  props: {
    entityType: { type: String, required: true }, // 'job' or 'profile'
    entityId: { type: Number, required: true },
    visible: { type: Boolean, default: true },
  },
  emits: ['close', 'modified'],
  template: `
    <div class="split-chat" :class="{ open: visible }">
      <div class="chat-header">
        <span class="text-xs font-mono font-bold text-white" style="letter-spacing:1.5px;">ATS BOT</span>
        <button @click="$emit('close')" class="md:hidden text-slate-400 hover:text-white" style="background:none;border:none;cursor:pointer;font-size:16px;">&#10005;</button>
      </div>
      <div class="chat-messages" ref="messagesEl">
        <div v-if="!messages.length && !loadingHistory" style="flex:1;"></div>
        <div v-for="(msg, i) in messages" :key="i" :class="['chat-bubble', msg.role]">
          <div v-html="formatMessage(msg.content)"></div>
        </div>
        <template v-if="activeTools.length">
          <div v-for="tool in activeTools" :key="tool.name" class="chat-tool-indicator">
            <span class="tool-dot"></span>
            {{ tool.label }}
          </div>
        </template>
        <div v-if="(sending || pendingResponse) && !activeTools.length" class="chat-bubble assistant">
          <span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
      </div>
      <div class="chat-input-area">
        <div v-if="!messages.length && !loadingHistory && !sending && !pendingResponse" style="margin-bottom:10px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:600;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:8px;">TRY ASKING</div>
          <div class="chat-starters">
            <button v-for="p in starterPrompts" :key="p" class="chat-starter-btn" @click="sendStarter(p)">
              <span style="flex:1;text-align:left;">{{ p }}</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.4;"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </button>
          </div>
        </div>
        <div style="display:flex;gap:8px;">
          <textarea v-model="input" @keydown.enter.exact.prevent="send" rows="1" class="input-field" style="flex:1;resize:none;min-height:38px;max-height:120px;overflow-y:auto;" placeholder="Ask about your resume..." @input="autoResize"></textarea>
          <button @click="send" class="btn-primary text-xs" style="padding:8px 14px;" :disabled="!input.trim() || sending || pendingResponse">SEND</button>
        </div>
        <div v-if="error" class="text-xs mt-2" style="color:var(--red);">{{ error }}</div>
        <button v-if="pendingResend && !pendingResponse && !sending" @click="resend" class="btn-primary text-xs mt-2" style="padding:6px 14px;background:var(--orange);color:#fff;">RESEND LAST MESSAGE</button>
      </div>
    </div>
  `,
  setup(props, { emit }) {
    const messages = ref([]);
    const input = ref('');
    const sending = ref(false);
    const error = ref(null);
    const loadingHistory = ref(false);
    const messagesEl = ref(null);
    const activeTools = ref([]);
    const pendingResponse = ref(false);
    const pollTimer = ref(null);
    const pollCount = ref(0);
    const pendingResend = ref(null);

    const POLL_INTERVAL = 5000;
    const POLL_MAX = 42;

    const JOB_STARTERS = [
      'Rewrite my first experience to better match this role',
      'Move my most relevant project to the top',
      'Which parts of my resume are weakest for this job?',
    ];
    const PROFILE_STARTERS = [
      'Add a new skill to my profile',
      'Rewrite my most recent role to sound more impactful',
      'Review my profile and suggest what\u2019s missing',
    ];
    const starterPrompts = computed(() => props.entityType === 'job' ? JOB_STARTERS : PROFILE_STARTERS);

    const sendStarter = (text) => {
      input.value = text;
      nextTick(() => send());
    };

    const scrollToBottom = () => {
      nextTick(() => {
        if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
      });
    };
    const autoResize = (e) => {
      const el = e.target;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    };

    const formatMessage = (text) => {
      if (!text) return '';
      // Escape HTML first
      let html = text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      // Process line-by-line for block elements
      const lines = html.split('\n');
      const out = [];
      let listTag = null; // 'ul' or 'ol' — tracks current open list type
      const closeList = () => { if (listTag) { out.push('</' + listTag + '>'); listTag = null; } };
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        // Headers: ### / ## / #
        if (/^#{1,4}\s/.test(line)) {
          closeList();
          const level = line.match(/^(#{1,4})/)[1].length;
          const sizes = { 1: '15px', 2: '14px', 3: '13px', 4: '12px' };
          const content = line.replace(/^#{1,4}\s+/, '');
          out.push(`<div style="font-weight:700;font-size:${sizes[level]};margin:10px 0 4px;color:var(--text-primary);">${content}</div>`);
          continue;
        }
        // Horizontal rule
        if (/^[-*_]{3,}\s*$/.test(line)) {
          closeList();
          out.push('<hr style="border:none;border-top:1px solid var(--card-border);margin:8px 0;">');
          continue;
        }
        // Unordered list: * or - at start
        if (/^\s*[*\-]\s+/.test(line)) {
          if (listTag !== 'ul') { closeList(); out.push('<ul style="margin:4px 0;padding-left:18px;">'); listTag = 'ul'; }
          out.push('<li>' + line.replace(/^\s*[*\-]\s+/, '') + '</li>');
          continue;
        }
        // Numbered list: 1. 2. etc
        if (/^\s*\d+\.\s+/.test(line)) {
          if (listTag !== 'ol') { closeList(); out.push('<ol style="margin:4px 0;padding-left:18px;">'); listTag = 'ol'; }
          out.push('<li>' + line.replace(/^\s*\d+\.\s+/, '') + '</li>');
          continue;
        }
        // Close list if we hit a non-list line
        closeList();
        // Empty line = paragraph break
        if (line.trim() === '') { out.push('<div style="height:8px;"></div>'); continue; }
        out.push(line + '<br>');
      }
      closeList();
      html = out.join('\n');
      // Inline formatting
      html = html
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:3px;font-size:12px;">$1</code>');
      return html;
    };

    const normalizeRole = (role) => role === 'model' ? 'assistant' : role;
    // Strip the [User's current time: ...] prefix that the backend prepends to user messages
    const stripTimePrefix = (text) => text ? text.replace(/^\[User's current time: [^\]]*\]\n?/, '') : text;

    // Separate history entries into chat messages and trailing in-progress tool calls.
    // Tool calls between user→assistant are completed (hidden). Tool calls after the
    // last user message with no assistant response are in-progress (shown as indicators).
    const processHistory = (raw) => {
      const msgs = [];
      const tools = [];
      for (const m of raw) {
        if (m.type === 'tool_call') {
          tools.push({ name: m.name, label: m.label });
        } else {
          tools.length = 0;  // text message — preceding tools are part of a completed turn
          msgs.push({
            ...m,
            role: normalizeRole(m.role),
            content: m.role === 'user' ? stripTimePrefix(m.content) : m.content,
          });
        }
      }
      // tools now holds only trailing tool_calls (after the last text message)
      const lastRole = msgs.length ? msgs[msgs.length - 1].role : null;
      return { msgs, pendingTools: lastRole === 'user' ? [...tools] : [] };
    };

    const loadHistory = async () => {
      loadingHistory.value = true;
      try {
        const prefix = props.entityType === 'job' ? 'jobs' : 'profiles';
        const data = await api.get(`/${prefix}/${props.entityId}/chat/history`);
        const { msgs, pendingTools } = processHistory(data.messages || []);
        messages.value = msgs;
        activeTools.value = pendingTools;
        scrollToBottom();
        // Detect interrupted chat: last message is user with no assistant reply.
        // Fire send() to reconnect to the in-flight agent's SSE stream (or start a new one).
        if (msgs.length > 0 && msgs[msgs.length - 1].role === 'user') {
          pendingResend.value = msgs[msgs.length - 1].content;
          send({ text: msgs[msgs.length - 1].content, skipPush: true });
        }
      } catch (e) {
        // History endpoint might 404 if no history — that's fine
      } finally {
        loadingHistory.value = false;
      }
    };

    const startPendingPoll = () => {
      if (pollTimer.value) return;
      pendingResponse.value = true;
      pollCount.value = 0;
      error.value = null;

      pollTimer.value = setInterval(async () => {
        pollCount.value++;
        if (pollCount.value > POLL_MAX) {
          stopPendingPoll();
          error.value = 'Response timed out.';
          return;
        }
        try {
          const prefix = props.entityType === 'job' ? 'jobs' : 'profiles';
          const data = await api.get(`/${prefix}/${props.entityId}/chat/history`);
          const { msgs, pendingTools } = processHistory(data.messages || []);
          // Update tool indicators as they appear in the ADK session
          activeTools.value = pendingTools;
          if (msgs.length && msgs[msgs.length - 1].role === 'assistant') {
            messages.value = msgs;
            scrollToBottom();
            stopPendingPoll();
            pendingResend.value = null;
          }
        } catch { /* retry next interval */ }
      }, POLL_INTERVAL);
    };

    const stopPendingPoll = () => {
      if (pollTimer.value) { clearInterval(pollTimer.value); pollTimer.value = null; }
      pendingResponse.value = false;
      pollCount.value = 0;
      activeTools.value = [];
    };

    const resend = () => {
      if (!pendingResend.value || sending.value) return;
      const text = pendingResend.value;
      error.value = null;
      pendingResend.value = null;
      send({ text, skipPush: true });
    };

    const send = async (opts = {}) => {
      const text = opts.text || input.value.trim();
      if (!text || sending.value) return;
      error.value = null;
      if (!opts.skipPush) { activeTools.value = []; pendingResend.value = null; }

      if (!opts.skipPush) {
        messages.value.push({ role: 'user', content: text });
      }
      if (!opts.text) input.value = '';
      sending.value = true;
      scrollToBottom();
      try {
        const prefix = props.entityType === 'job' ? 'jobs' : 'profiles';
        const now = new Date();
        const token = localStorage.getItem('token');
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch(`/${prefix}/${props.entityId}/chat`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            message: text,
            client_datetime: now.toISOString(),
            client_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
          }),
        });
        if (!res.ok) {
          if (res.status === 409) {
            pendingResend.value = text;
            startPendingPoll();
            return;
          }
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        // Parse SSE stream
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let gotResponse = false;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          // Process complete SSE events (lines starting with "data: ")
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep incomplete line in buffer
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const jsonStr = line.slice(6);
            if (!jsonStr) continue;
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'tool_call') {
                // Add tool indicator (avoid duplicates)
                if (!activeTools.value.find(t => t.name === event.name)) {
                  activeTools.value.push({ name: event.name, label: event.label });
                  scrollToBottom();
                }
              } else if (event.type === 'response') {
                gotResponse = true;
                activeTools.value = [];
                sending.value = false;  // done — stream may stay open for PDF recompile
                messages.value.push({ role: 'assistant', content: event.response });
                scrollToBottom();
                if (event.resume_modified) {
                  const payload = {};
                  if (props.entityType === 'job' && event.custom_resume_data) {
                    payload.custom_resume_data = event.custom_resume_data;
                  } else if (props.entityType === 'profile' && event.resume_info) {
                    payload.resume_info = event.resume_info;
                  }
                  emit('modified', payload);
                }
              } else if (event.type === 'error') {
                error.value = event.message || 'Something went wrong';
                sending.value = false;
                gotResponse = true; // prevent double error
              }
            } catch (parseErr) { /* skip malformed JSON */ }
          }
        }
        // If stream ended without a response event, show an error
        if (!gotResponse) {
          throw new Error('No response received — please try again');
        }
      } catch (e) {
        error.value = e.message || 'Failed to send message';
      } finally {
        activeTools.value = [];
        sending.value = false;
      }
    };

    onMounted(loadHistory);
    onUnmounted(() => stopPendingPoll());

    return { messages, input, sending, error, loadingHistory, messagesEl, activeTools, starterPrompts, scrollToBottom, autoResize, formatMessage, send, sendStarter, pendingResponse, pendingResend, resend };
  },
};

const routes = [
  { path: '/login', component: LoginPage, meta: { public: true } },
  {
    path: '/',
    component: AppLayout,
    children: [
      { path: '', redirect: '/dashboard' },
      { path: 'dashboard', component: DashboardPage },
      { path: 'profiles', component: ProfileListPage },
      { path: 'profiles/new', component: ProfileUploadPage },
      { path: 'profiles/view', component: ProfileDetailPage },
      { path: 'jobs', component: JobListPage },
      { path: 'jobs/new', component: JobCreatePage },
      { path: 'jobs/view', component: JobDetailPage },
      { path: 'roast', component: RoastPage },
      { path: 'credits', component: CreditHistoryPage },
      { path: 'settings/ai', component: AISettingsPage },
      { path: 'admin', component: AdminPage, meta: { requiresSuperAdmin: true } },
    ],
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

// Navigation guard
router.beforeEach(async (to, from) => {
  const auth = useAuthStore();

  // Handle OAuth callback: store token and redirect before page renders
  if (to.path === '/login' && to.query.access_token) {
    localStorage.setItem('token', to.query.access_token);
    await auth.fetchMe();
    if (auth.user) return '/dashboard';
    // Token was invalid — clean up and show login (strip token from URL)
    localStorage.removeItem('token');
    return '/login';
  }

  // If user isn't loaded yet but has a stored token, try fetching
  if (!auth.user && localStorage.getItem('token')) {
    await auth.fetchMe();
  }

  // Redirect logged-in users away from login page
  if (to.path === '/login' && auth.user) return '/dashboard';
  // Block unauthenticated users from protected pages
  if (!to.meta.public && !auth.user) return '/login';
  // Block non-super-admin from admin routes
  if (to.meta.requiresSuperAdmin && !auth.isSuperAdmin) return '/dashboard';
  return true;
});

// ================================================================
// APP INIT
// ================================================================
const app = createApp({
  template: '<router-view />',
});

app.component('TopHeader', TopHeader);
app.component('ChatPanel', ChatPanel);
app.use(createPinia());
app.use(router);
app.mount('#app');
