class TicketAccessPolicy
  class Unauthenticated < StandardError; end
  attr_reader :membership

  def initialize(session)
    @membership = Membership.find_by(id: session[:membership_id], user_id: session[:user_id])
    raise Unauthenticated unless @membership
  end

  def tickets
    scope = Ticket.where(organization_id: membership.organization_id)
    case membership.role
    when 'admin', 'agent' then scope
    when 'end_customer' then scope.where(user_id: membership.user_id)
    else scope.none
    end
  end

  def ticket(id)
    tickets.find(id)
  end

  def attachment(ticket_id, id)
    ticket(ticket_id).ticket_attachments.where(organization_id: membership.organization_id).find(id)
  end
end
