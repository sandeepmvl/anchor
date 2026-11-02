module Admin
  class AiCostsController < ActionController::Base
    def index
      tenant_id = Integer(params[:organization_id], exception: false)
      user_id = session[:user_id]
      return head :unauthorized unless user_id
      membership = Membership.find_by(user_id: user_id, organization_id: tenant_id, role: 'admin')
      return head :forbidden unless membership
      @window = params[:window].presence || Time.now.utc.strftime('%Y-%m')
      return head :unprocessable_entity unless /\A\d{4}-(0[1-9]|1[0-2])\z/.match?(@window)
      @config = AiBudgetCounter.configuration
      if params[:comparison] == 'export'
        rows = InferenceComparisonExporter.new.export(
          user_id: user_id, organization_id: tenant_id, window: @window,
          cost_unit: @config.fetch('unit'))
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Disposition'] = 'attachment; filename="hosting_usage.jsonl"'
        return render plain: rows.map { |row| JSON.generate(row) }.join("\n") + "\n",
                      content_type: 'application/x-ndjson'
      end
      @counters = AiBudgetCounter.dashboard(tenant_id: tenant_id, window: @window)
      render :index
    end
  end
end
